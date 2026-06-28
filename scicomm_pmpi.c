/*
 * SciComm PMPI metadata recorder.
 *
 * This library is attached with LD_PRELOAD and writes one JSONL trace shard per
 * MPI rank. It records descriptors only: operation, sizes, peers, communicator
 * ids, placement, and timing. It never records payload data.
 */

#include <mpi.h>

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
  MPI_Comm comm;
  int id;
  int size;
} CommInfo;

typedef struct {
  MPI_Request request;
  int active;
  char op[32];
  int peer;
  int tag;
  int comm_id;
  int comm_size;
  long long bytes;
  long long send_bytes;
  long long recv_bytes;
  long long send_max_bytes;
  long long recv_max_bytes;
  long long send_min_nonzero_bytes;
  long long recv_min_nonzero_bytes;
  int send_nonzero;
  int recv_nonzero;
  double start_us;
  char dtype[32];
  char buffer_location[16];
} RequestInfo;

typedef struct {
  long long total_bytes;
  long long max_bytes;
  long long min_nonzero_bytes;
  int nonzero;
} CountStats;

static FILE *g_out = NULL;
static int g_rank = -1;
static int g_size = -1;
static int g_local_rank = -1;
static char g_host[256] = "unknown";
static char g_phase[128] = "unknown";
static char g_buffer_location[16] = "unknown";
static int g_callsite_enabled = 0;
static int g_flush_every = 0;
static long long g_seq = 0;
static double g_t0 = 0.0;
static int g_ready = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static CommInfo *g_comms = NULL;
static int g_comm_count = 0;
static int g_comm_cap = 0;

static RequestInfo *g_requests = NULL;
static int g_request_count = 0;
static int g_request_cap = 0;

void scicomm_set_phase(const char *phase) {
  pthread_mutex_lock(&g_lock);
  if (phase && phase[0]) {
    snprintf(g_phase, sizeof(g_phase), "%s", phase);
  } else {
    snprintf(g_phase, sizeof(g_phase), "%s", "unknown");
  }
  pthread_mutex_unlock(&g_lock);
}

static int env_truthy(const char *name) {
  const char *value = getenv(name);
  return value && value[0] && strcmp(value, "0") != 0 && strcmp(value, "false") != 0;
}

static void read_env(void) {
  const char *phase = getenv("SCICOMM_PHASE");
  const char *location = getenv("SCICOMM_BUFFER_LOCATION");
  const char *flush_every = getenv("SCICOMM_FLUSH_EVERY");

  if (phase && phase[0]) {
    snprintf(g_phase, sizeof(g_phase), "%s", phase);
  }
  if (location && location[0]) {
    snprintf(g_buffer_location, sizeof(g_buffer_location), "%s", location);
  }
  if (flush_every && flush_every[0]) {
    g_flush_every = atoi(flush_every);
    if (g_flush_every < 0) {
      g_flush_every = 0;
    }
  }
  g_callsite_enabled = env_truthy("SCICOMM_CALLSITE");
}

static double now_us(void) {
  if (!g_ready) {
    return 0.0;
  }
  return (PMPI_Wtime() - g_t0) * 1000000.0;
}

static void json_string(FILE *out, const char *value) {
  const unsigned char *p = (const unsigned char *) (value ? value : "");
  fputc('"', out);
  while (*p) {
    switch (*p) {
      case '\\':
        fputs("\\\\", out);
        break;
      case '"':
        fputs("\\\"", out);
        break;
      case '\n':
        fputs("\\n", out);
        break;
      case '\r':
        fputs("\\r", out);
        break;
      case '\t':
        fputs("\\t", out);
        break;
      default:
        if (*p < 0x20) {
          fprintf(out, "\\u%04x", *p);
        } else {
          fputc(*p, out);
        }
    }
    p++;
  }
  fputc('"', out);
}

static void log_begin(const char *event_name) {
  if (!g_out) {
    return;
  }
  fprintf(g_out,
          "{\"schema\":\"scicomm.raw.v1\",\"event\":\"%s\",\"seq\":%lld,"
          "\"rank\":%d,\"size\":%d,\"time_us\":%.3f,\"host\":",
          event_name, ++g_seq, g_rank, g_size, now_us());
  json_string(g_out, g_host);
  fprintf(g_out, ",\"local_rank\":%d", g_local_rank);
}

static void log_end(void *callsite) {
  if (!g_out) {
    return;
  }
  fprintf(g_out, ",\"phase\":");
  json_string(g_out, g_phase);
  if (g_callsite_enabled && callsite) {
    fprintf(g_out, ",\"callsite\":\"%p\"", callsite);
  }
  fputs("}\n", g_out);
  if (g_flush_every > 0 && (g_seq % g_flush_every) == 0) {
    fflush(g_out);
  }
}

static const char *dtype_name(MPI_Datatype datatype) {
  if (datatype == MPI_CHAR) return "char";
  if (datatype == MPI_SIGNED_CHAR) return "int8";
  if (datatype == MPI_UNSIGNED_CHAR) return "uint8";
  if (datatype == MPI_SHORT) return "int16";
  if (datatype == MPI_UNSIGNED_SHORT) return "uint16";
  if (datatype == MPI_INT) return "int32";
  if (datatype == MPI_UNSIGNED) return "uint32";
  if (datatype == MPI_LONG) return "long";
  if (datatype == MPI_UNSIGNED_LONG) return "ulong";
  if (datatype == MPI_LONG_LONG) return "int64";
  if (datatype == MPI_UNSIGNED_LONG_LONG) return "uint64";
  if (datatype == MPI_FLOAT) return "float32";
  if (datatype == MPI_DOUBLE) return "float64";
  if (datatype == MPI_LONG_DOUBLE) return "long_double";
  if (datatype == MPI_BYTE) return "byte";
  return "unknown";
}

static const char *op_name(MPI_Op op) {
  if (op == MPI_SUM) return "sum";
  if (op == MPI_MAX) return "max";
  if (op == MPI_MIN) return "min";
  if (op == MPI_PROD) return "prod";
  if (op == MPI_LAND) return "land";
  if (op == MPI_BAND) return "band";
  if (op == MPI_LOR) return "lor";
  if (op == MPI_BOR) return "bor";
  if (op == MPI_LXOR) return "lxor";
  if (op == MPI_BXOR) return "bxor";
  if (op == MPI_MAXLOC) return "maxloc";
  if (op == MPI_MINLOC) return "minloc";
  return "user_or_unknown";
}

static long long datatype_bytes(MPI_Datatype datatype, int count) {
  int size = 0;
  if (count <= 0) {
    return 0;
  }
  if (PMPI_Type_size(datatype, &size) != MPI_SUCCESS || size < 0) {
    return 0;
  }
  return (long long) count * (long long) size;
}

static CountStats count_stats(const int *counts, MPI_Datatype datatype, int n) {
  CountStats stats;
  int i;
  memset(&stats, 0, sizeof(stats));
  stats.min_nonzero_bytes = 0;
  if (!counts || n <= 0) {
    return stats;
  }
  for (i = 0; i < n; i++) {
    long long bytes = datatype_bytes(datatype, counts[i]);
    stats.total_bytes += bytes;
    if (bytes > 0) {
      stats.nonzero++;
      if (bytes > stats.max_bytes) {
        stats.max_bytes = bytes;
      }
      if (stats.min_nonzero_bytes == 0 || bytes < stats.min_nonzero_bytes) {
        stats.min_nonzero_bytes = bytes;
      }
    }
  }
  return stats;
}

static int ensure_comm_capacity(void) {
  if (g_comm_count < g_comm_cap) {
    return 1;
  }
  int next_cap = g_comm_cap ? g_comm_cap * 2 : 32;
  CommInfo *next = (CommInfo *) realloc(g_comms, (size_t) next_cap * sizeof(CommInfo));
  if (!next) {
    return 0;
  }
  g_comms = next;
  g_comm_cap = next_cap;
  return 1;
}

static int comm_id(MPI_Comm comm) {
  int i;
  int size = -1;
  if (comm == MPI_COMM_NULL) {
    return -1;
  }
  for (i = 0; i < g_comm_count; i++) {
    if (g_comms[i].comm == comm) {
      return g_comms[i].id;
    }
  }
  if (!ensure_comm_capacity()) {
    return -1;
  }
  PMPI_Comm_size(comm, &size);
  g_comms[g_comm_count].comm = comm;
  g_comms[g_comm_count].id = g_comm_count;
  g_comms[g_comm_count].size = size;
  g_comm_count++;
  return g_comm_count - 1;
}

static int comm_size_by_id(int id) {
  if (id < 0 || id >= g_comm_count) {
    return -1;
  }
  return g_comms[id].size;
}

static int ensure_request_capacity(void) {
  if (g_request_count < g_request_cap) {
    return 1;
  }
  int next_cap = g_request_cap ? g_request_cap * 2 : 1024;
  RequestInfo *next = (RequestInfo *) realloc(g_requests, (size_t) next_cap * sizeof(RequestInfo));
  if (!next) {
    return 0;
  }
  g_requests = next;
  g_request_cap = next_cap;
  return 1;
}

static int find_request_index(MPI_Request request) {
  int i;
  if (request == MPI_REQUEST_NULL) {
    return -1;
  }
  for (i = 0; i < g_request_count; i++) {
    if (g_requests[i].active && g_requests[i].request == request) {
      return i;
    }
  }
  return -1;
}

static void remember_request(MPI_Request request, const char *op, int peer, int tag,
                             int cid, long long bytes, MPI_Datatype datatype) {
  RequestInfo *info;
  if (request == MPI_REQUEST_NULL || !ensure_request_capacity()) {
    return;
  }
  info = &g_requests[g_request_count++];
  memset(info, 0, sizeof(*info));
  info->request = request;
  info->active = 1;
  snprintf(info->op, sizeof(info->op), "%s", op ? op : "unknown");
  info->peer = peer;
  info->tag = tag;
  info->comm_id = cid;
  info->comm_size = comm_size_by_id(cid);
  info->bytes = bytes;
  info->start_us = now_us();
  snprintf(info->dtype, sizeof(info->dtype), "%s", dtype_name(datatype));
  snprintf(info->buffer_location, sizeof(info->buffer_location), "%s", g_buffer_location);
}

static int snapshot_request(MPI_Request request, RequestInfo *out, int *index_out) {
  int idx = find_request_index(request);
  if (idx < 0) {
    return 0;
  }
  if (out) {
    *out = g_requests[idx];
  }
  if (index_out) {
    *index_out = idx;
  }
  return 1;
}

static void deactivate_request_index(int idx) {
  if (idx >= 0 && idx < g_request_count) {
    g_requests[idx].active = 0;
  }
}

static void set_request_transfer_stats(MPI_Request request, long long send_bytes,
                                       long long recv_bytes, int send_nonzero,
                                       int recv_nonzero, long long send_max_bytes,
                                       long long recv_max_bytes,
                                       long long send_min_nonzero_bytes,
                                       long long recv_min_nonzero_bytes) {
  int idx = find_request_index(request);
  RequestInfo *info;
  if (idx < 0) {
    return;
  }
  info = &g_requests[idx];
  info->send_bytes = send_bytes;
  info->recv_bytes = recv_bytes;
  info->send_nonzero = send_nonzero;
  info->recv_nonzero = recv_nonzero;
  info->send_max_bytes = send_max_bytes;
  info->recv_max_bytes = recv_max_bytes;
  info->send_min_nonzero_bytes = send_min_nonzero_bytes;
  info->recv_min_nonzero_bytes = recv_min_nonzero_bytes;
}

static void log_request_vector_stats(const RequestInfo *info) {
  if (!g_out) {
    return;
  }
  if (info->send_bytes || info->recv_bytes || info->send_nonzero || info->recv_nonzero) {
    fprintf(g_out,
            ",\"send_bytes\":%lld,\"recv_bytes\":%lld,"
            "\"send_nonzero\":%d,\"recv_nonzero\":%d,"
            "\"send_max_bytes\":%lld,\"recv_max_bytes\":%lld,"
            "\"send_min_nonzero_bytes\":%lld,\"recv_min_nonzero_bytes\":%lld",
            info->send_bytes, info->recv_bytes,
            info->send_nonzero, info->recv_nonzero,
            info->send_max_bytes, info->recv_max_bytes,
            info->send_min_nonzero_bytes, info->recv_min_nonzero_bytes);
  }
}

static void log_request_start(const RequestInfo *info, void *callsite) {
  pthread_mutex_lock(&g_lock);
  log_begin("request_start");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"%s\",\"comm_id\":%d,\"comm_size\":%d,"
            "\"peer\":%d,\"tag\":%d,\"bytes\":%lld,\"dtype\":",
            info->op, info->comm_id, info->comm_size, info->peer, info->tag, info->bytes);
    json_string(g_out, info->dtype);
    log_request_vector_stats(info);
    fprintf(g_out, ",\"buffer_location\":");
    json_string(g_out, info->buffer_location);
  }
  log_end(callsite);
  pthread_mutex_unlock(&g_lock);
}

static void log_request_complete(const RequestInfo *info, const char *completion,
                                 double wait_duration_us, void *callsite) {
  double elapsed = now_us() - info->start_us;
  pthread_mutex_lock(&g_lock);
  log_begin("request_complete");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"%s\",\"completion\":\"%s\",\"comm_id\":%d,\"comm_size\":%d,"
            "\"peer\":%d,\"tag\":%d,\"bytes\":%lld,\"dtype\":",
            info->op, completion, info->comm_id, info->comm_size, info->peer, info->tag, info->bytes);
    json_string(g_out, info->dtype);
    log_request_vector_stats(info);
    fprintf(g_out, ",\"elapsed_us\":%.3f,\"wait_duration_us\":%.3f,\"buffer_location\":",
            elapsed, wait_duration_us);
    json_string(g_out, info->buffer_location);
  }
  log_end(callsite);
  pthread_mutex_unlock(&g_lock);
}

static void log_unknown_wait(const char *completion, double duration_us, void *callsite) {
  pthread_mutex_lock(&g_lock);
  log_begin("request_complete");
  if (g_out) {
    fprintf(g_out, ",\"op\":\"unknown\",\"completion\":\"%s\",\"wait_duration_us\":%.3f",
            completion, duration_us);
  }
  log_end(callsite);
  pthread_mutex_unlock(&g_lock);
}

static void init_output(void) {
  char path[PATH_MAX];
  const char *dir = getenv("SCICOMM_TRACE_DIR");
  const char *prefix = getenv("SCICOMM_TRACE_PREFIX");
  if (!dir || !dir[0]) {
    dir = ".";
  }
  if (!prefix || !prefix[0]) {
    prefix = "scicomm";
  }
  if (mkdir(dir, 0777) != 0 && errno != EEXIST) {
    dir = ".";
  }
  snprintf(path, sizeof(path), "%s/%s.rank%06d.jsonl", dir, prefix, g_rank);
  g_out = fopen(path, "w");
  if (g_out) {
    setvbuf(g_out, NULL, _IOFBF, 1 << 20);
  }
}

static void detect_local_rank(void) {
  const char *env_names[] = {
      "SLURM_LOCALID",
      "OMPI_COMM_WORLD_LOCAL_RANK",
      "MV2_COMM_WORLD_LOCAL_RANK",
      "MPI_LOCALRANKID",
      "PMI_LOCAL_RANK",
      NULL,
  };
  int i;
  for (i = 0; env_names[i]; i++) {
    const char *value = getenv(env_names[i]);
    if (value && value[0]) {
      g_local_rank = atoi(value);
      return;
    }
  }

#if MPI_VERSION >= 3
  {
    MPI_Comm local_comm = MPI_COMM_NULL;
    int local_rank = -1;
    if (PMPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank,
                             MPI_INFO_NULL, &local_comm) == MPI_SUCCESS) {
      PMPI_Comm_rank(local_comm, &local_rank);
      PMPI_Comm_free(&local_comm);
      g_local_rank = local_rank;
    }
  }
#endif
}

static void after_init(void) {
  read_env();
  gethostname(g_host, sizeof(g_host));
  g_host[sizeof(g_host) - 1] = '\0';
  PMPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &g_size);
  detect_local_rank();
  g_t0 = PMPI_Wtime();
  g_ready = 1;
  init_output();
  comm_id(MPI_COMM_WORLD);

  pthread_mutex_lock(&g_lock);
  log_begin("rank_info");
  if (g_out) {
    fprintf(g_out, ",\"comm_id\":0,\"buffer_location_default\":");
    json_string(g_out, g_buffer_location);
  }
  log_end(NULL);
  pthread_mutex_unlock(&g_lock);
}

int MPI_Init(int *argc, char ***argv) {
  int rc = PMPI_Init(argc, argv);
  if (rc == MPI_SUCCESS) {
    after_init();
  }
  return rc;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
  int rc = PMPI_Init_thread(argc, argv, required, provided);
  if (rc == MPI_SUCCESS) {
    after_init();
  }
  return rc;
}

int MPI_Finalize(void) {
  int rc;
  pthread_mutex_lock(&g_lock);
  log_begin("finalize");
  log_end(NULL);
  if (g_out) {
    fflush(g_out);
  }
  pthread_mutex_unlock(&g_lock);

  rc = PMPI_Finalize();

  pthread_mutex_lock(&g_lock);
  if (g_out) {
    fclose(g_out);
    g_out = NULL;
  }
  free(g_comms);
  g_comms = NULL;
  g_comm_count = 0;
  g_comm_cap = 0;
  free(g_requests);
  g_requests = NULL;
  g_request_count = 0;
  g_request_cap = 0;
  g_ready = 0;
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
  double start = now_us();
  int parent_id;
  int new_id = -1;
  int rc = PMPI_Comm_dup(comm, newcomm);
  if (rc == MPI_SUCCESS && newcomm) {
    parent_id = comm_id(comm);
    new_id = comm_id(*newcomm);
    pthread_mutex_lock(&g_lock);
    log_begin("comm_create");
    if (g_out) {
      fprintf(g_out, ",\"op\":\"comm_dup\",\"parent_comm_id\":%d,\"comm_id\":%d,"
              "\"comm_size\":%d,\"duration_us\":%.3f",
              parent_id, new_id, comm_size_by_id(new_id), now_us() - start);
    }
    log_end(__builtin_return_address(0));
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
  double start = now_us();
  int parent_id;
  int new_id = -1;
  int rc = PMPI_Comm_split(comm, color, key, newcomm);
  if (rc == MPI_SUCCESS && newcomm && *newcomm != MPI_COMM_NULL) {
    parent_id = comm_id(comm);
    new_id = comm_id(*newcomm);
    pthread_mutex_lock(&g_lock);
    log_begin("comm_create");
    if (g_out) {
      fprintf(g_out, ",\"op\":\"comm_split\",\"parent_comm_id\":%d,\"comm_id\":%d,"
              "\"comm_size\":%d,\"color\":%d,\"key\":%d,\"duration_us\":%.3f",
              parent_id, new_id, comm_size_by_id(new_id), color, key, now_us() - start);
    }
    log_end(__builtin_return_address(0));
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

int MPI_Comm_free(MPI_Comm *comm) {
  int cid = (comm && *comm != MPI_COMM_NULL) ? comm_id(*comm) : -1;
  double start = now_us();
  int rc = PMPI_Comm_free(comm);
  pthread_mutex_lock(&g_lock);
  log_begin("comm_free");
  if (g_out) {
    fprintf(g_out, ",\"comm_id\":%d,\"duration_us\":%.3f", cid, now_us() - start);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  double start = now_us();
  int cid = comm_id(comm);
  long long bytes = datatype_bytes(datatype, count);
  int rc = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  pthread_mutex_lock(&g_lock);
  log_begin("collective");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"allreduce\",\"comm_id\":%d,\"comm_size\":%d,"
            "\"count\":%d,\"dtype\":",
            cid, comm_size_by_id(cid), count);
    json_string(g_out, dtype_name(datatype));
    fprintf(g_out, ",\"bytes\":%lld,\"reduction\":", bytes);
    json_string(g_out, op_name(op));
    fprintf(g_out, ",\"duration_us\":%.3f,\"buffer_location\":", now_us() - start);
    json_string(g_out, g_buffer_location);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  double start = now_us();
  int cid = comm_id(comm);
  long long bytes = datatype_bytes(datatype, count);
  int rc = PMPI_Bcast(buffer, count, datatype, root, comm);
  pthread_mutex_lock(&g_lock);
  log_begin("collective");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"bcast\",\"comm_id\":%d,\"comm_size\":%d,\"root\":%d,"
            "\"count\":%d,\"dtype\":",
            cid, comm_size_by_id(cid), root, count);
    json_string(g_out, dtype_name(datatype));
    fprintf(g_out, ",\"bytes\":%lld,\"duration_us\":%.3f,\"buffer_location\":",
            bytes, now_us() - start);
    json_string(g_out, g_buffer_location);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm) {
  double start = now_us();
  int cid = comm_id(comm);
  int csize = comm_size_by_id(cid);
  long long send_bytes = datatype_bytes(sendtype, sendcount);
  long long recv_one = datatype_bytes(recvtype, recvcount);
  long long recv_bytes = recv_one * (long long) csize;
  int recv_nonzero = recv_one > 0 ? csize : 0;
  int rc = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  pthread_mutex_lock(&g_lock);
  log_begin("collective");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"allgather\",\"comm_id\":%d,\"comm_size\":%d,"
            "\"send_count\":%d,\"recv_count\":%d,\"send_dtype\":",
            cid, csize, sendcount, recvcount);
    json_string(g_out, dtype_name(sendtype));
    fprintf(g_out, ",\"recv_dtype\":");
    json_string(g_out, dtype_name(recvtype));
    fprintf(g_out,
            ",\"send_bytes\":%lld,\"recv_bytes\":%lld,"
            "\"recv_nonzero\":%d,\"recv_max_bytes\":%lld,"
            "\"recv_min_nonzero_bytes\":%lld,"
            "\"duration_us\":%.3f,\"buffer_location\":",
            send_bytes, recv_bytes, recv_nonzero, recv_one,
            recv_nonzero ? recv_one : 0, now_us() - start);
    json_string(g_out, g_buffer_location);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, const int recvcounts[], const int displs[],
                   MPI_Datatype recvtype, MPI_Comm comm) {
  double start = now_us();
  int cid = comm_id(comm);
  int csize = comm_size_by_id(cid);
  long long send_bytes = datatype_bytes(sendtype, sendcount);
  CountStats recv_stats = count_stats(recvcounts, recvtype, csize);
  int rc = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                           displs, recvtype, comm);
  pthread_mutex_lock(&g_lock);
  log_begin("collective");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"allgatherv\",\"comm_id\":%d,\"comm_size\":%d,"
            "\"send_count\":%d,\"send_dtype\":",
            cid, csize, sendcount);
    json_string(g_out, dtype_name(sendtype));
    fprintf(g_out, ",\"recv_dtype\":");
    json_string(g_out, dtype_name(recvtype));
    fprintf(g_out,
            ",\"send_bytes\":%lld,\"recv_bytes\":%lld,"
            "\"recv_nonzero\":%d,\"recv_max_bytes\":%lld,"
            "\"recv_min_nonzero_bytes\":%lld,"
            "\"duration_us\":%.3f,\"buffer_location\":",
            send_bytes, recv_stats.total_bytes,
            recv_stats.nonzero, recv_stats.max_bytes,
            recv_stats.min_nonzero_bytes, now_us() - start);
    json_string(g_out, g_buffer_location);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Barrier(MPI_Comm comm) {
  double start = now_us();
  int cid = comm_id(comm);
  int rc = PMPI_Barrier(comm);
  pthread_mutex_lock(&g_lock);
  log_begin("collective");
  if (g_out) {
    fprintf(g_out, ",\"op\":\"barrier\",\"comm_id\":%d,\"comm_size\":%d,"
            "\"duration_us\":%.3f",
            cid, comm_size_by_id(cid), now_us() - start);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
  double start = now_us();
  int cid = comm_id(comm);
  int csize = comm_size_by_id(cid);
  long long send_bytes = datatype_bytes(sendtype, sendcount) * (long long) csize;
  long long recv_bytes = datatype_bytes(recvtype, recvcount) * (long long) csize;
  int rc = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  pthread_mutex_lock(&g_lock);
  log_begin("collective");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"alltoall\",\"comm_id\":%d,\"comm_size\":%d,"
            "\"send_count\":%d,\"recv_count\":%d,\"send_dtype\":",
            cid, csize, sendcount, recvcount);
    json_string(g_out, dtype_name(sendtype));
    fprintf(g_out, ",\"recv_dtype\":");
    json_string(g_out, dtype_name(recvtype));
    fprintf(g_out, ",\"send_bytes\":%lld,\"recv_bytes\":%lld,\"duration_us\":%.3f,"
            "\"buffer_location\":",
            send_bytes, recv_bytes, now_us() - start);
    json_string(g_out, g_buffer_location);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                  MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                  const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm) {
  double start = now_us();
  int cid = comm_id(comm);
  int csize = comm_size_by_id(cid);
  CountStats send_stats = count_stats(sendcounts, sendtype, csize);
  CountStats recv_stats = count_stats(recvcounts, recvtype, csize);
  int rc = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf,
                          recvcounts, rdispls, recvtype, comm);
  pthread_mutex_lock(&g_lock);
  log_begin("collective");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"alltoallv\",\"comm_id\":%d,\"comm_size\":%d,"
            "\"send_dtype\":",
            cid, csize);
    json_string(g_out, dtype_name(sendtype));
    fprintf(g_out, ",\"recv_dtype\":");
    json_string(g_out, dtype_name(recvtype));
    fprintf(g_out,
            ",\"send_bytes\":%lld,\"recv_bytes\":%lld,"
            "\"send_nonzero\":%d,\"recv_nonzero\":%d,"
            "\"send_max_bytes\":%lld,\"recv_max_bytes\":%lld,"
            "\"send_min_nonzero_bytes\":%lld,\"recv_min_nonzero_bytes\":%lld,"
            "\"duration_us\":%.3f,\"buffer_location\":",
            send_stats.total_bytes, recv_stats.total_bytes,
            send_stats.nonzero, recv_stats.nonzero,
            send_stats.max_bytes, recv_stats.max_bytes,
            send_stats.min_nonzero_bytes, recv_stats.min_nonzero_bytes,
            now_us() - start);
    json_string(g_out, g_buffer_location);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
             MPI_Comm comm) {
  double start = now_us();
  int cid = comm_id(comm);
  long long bytes = datatype_bytes(datatype, count);
  int rc = PMPI_Send(buf, count, datatype, dest, tag, comm);
  pthread_mutex_lock(&g_lock);
  log_begin("p2p");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"send\",\"comm_id\":%d,\"comm_size\":%d,\"peer\":%d,"
            "\"tag\":%d,\"count\":%d,\"dtype\":",
            cid, comm_size_by_id(cid), dest, tag, count);
    json_string(g_out, dtype_name(datatype));
    fprintf(g_out, ",\"bytes\":%lld,\"duration_us\":%.3f,\"buffer_location\":",
            bytes, now_us() - start);
    json_string(g_out, g_buffer_location);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status) {
  double start = now_us();
  int cid = comm_id(comm);
  long long expected_bytes = datatype_bytes(datatype, count);
  int actual_source = source;
  int actual_tag = tag;
  int actual_count = -1;
  long long actual_bytes = -1;
  int rc = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
  if (rc == MPI_SUCCESS && status != MPI_STATUS_IGNORE) {
    actual_source = status->MPI_SOURCE;
    actual_tag = status->MPI_TAG;
    if (PMPI_Get_count(status, datatype, &actual_count) == MPI_SUCCESS && actual_count >= 0) {
      actual_bytes = datatype_bytes(datatype, actual_count);
    }
  }
  pthread_mutex_lock(&g_lock);
  log_begin("p2p");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"recv\",\"comm_id\":%d,\"comm_size\":%d,\"peer\":%d,"
            "\"requested_peer\":%d,\"tag\":%d,\"requested_tag\":%d,"
            "\"count\":%d,\"actual_count\":%d,\"dtype\":",
            cid, comm_size_by_id(cid), actual_source, source, actual_tag, tag, count, actual_count);
    json_string(g_out, dtype_name(datatype));
    fprintf(g_out,
            ",\"expected_bytes\":%lld,\"bytes\":%lld,\"duration_us\":%.3f,"
            "\"buffer_location\":",
            expected_bytes, actual_bytes >= 0 ? actual_bytes : expected_bytes, now_us() - start);
    json_string(g_out, g_buffer_location);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request *request) {
  int cid = comm_id(comm);
  long long bytes = datatype_bytes(datatype, count);
  RequestInfo info;
  int rc = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
  if (rc == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL) {
    remember_request(*request, "isend", dest, tag, cid, bytes, datatype);
    if (snapshot_request(*request, &info, NULL)) {
      log_request_start(&info, __builtin_return_address(0));
    }
  }
  return rc;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request) {
  int cid = comm_id(comm);
  long long bytes = datatype_bytes(datatype, count);
  RequestInfo info;
  int rc = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
  if (rc == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL) {
    remember_request(*request, "irecv", source, tag, cid, bytes, datatype);
    if (snapshot_request(*request, &info, NULL)) {
      log_request_start(&info, __builtin_return_address(0));
    }
  }
  return rc;
}

#if MPI_VERSION >= 3
int MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count,
                   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                   MPI_Request *request) {
  int cid = comm_id(comm);
  long long bytes = datatype_bytes(datatype, count);
  RequestInfo info;
  int rc = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
  if (rc == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL) {
    remember_request(*request, "iallreduce", -1, -1, cid, bytes, datatype);
    if (snapshot_request(*request, &info, NULL)) {
      snprintf(info.op, sizeof(info.op), "%s", "iallreduce");
      log_request_start(&info, __builtin_return_address(0));
    }
  }
  (void) op;
  return rc;
}

int MPI_Iallgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm, MPI_Request *request) {
  int cid = comm_id(comm);
  int csize = comm_size_by_id(cid);
  long long send_bytes = datatype_bytes(sendtype, sendcount);
  long long recv_one = datatype_bytes(recvtype, recvcount);
  long long recv_bytes = recv_one * (long long) csize;
  int recv_nonzero = recv_one > 0 ? csize : 0;
  RequestInfo info;
  int rc = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                           recvtype, comm, request);
  if (rc == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL) {
    remember_request(*request, "iallgather", -1, -1, cid, recv_bytes, recvtype);
    set_request_transfer_stats(*request, send_bytes, recv_bytes,
                               send_bytes > 0 ? 1 : 0, recv_nonzero,
                               send_bytes, recv_one,
                               send_bytes > 0 ? send_bytes : 0,
                               recv_nonzero ? recv_one : 0);
    if (snapshot_request(*request, &info, NULL)) {
      log_request_start(&info, __builtin_return_address(0));
    }
  }
  return rc;
}

int MPI_Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, const int recvcounts[], const int displs[],
                    MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) {
  int cid = comm_id(comm);
  int csize = comm_size_by_id(cid);
  long long send_bytes = datatype_bytes(sendtype, sendcount);
  CountStats recv_stats = count_stats(recvcounts, recvtype, csize);
  RequestInfo info;
  int rc = PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                            displs, recvtype, comm, request);
  if (rc == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL) {
    remember_request(*request, "iallgatherv", -1, -1, cid, recv_stats.total_bytes, recvtype);
    set_request_transfer_stats(*request, send_bytes, recv_stats.total_bytes,
                               send_bytes > 0 ? 1 : 0, recv_stats.nonzero,
                               send_bytes, recv_stats.max_bytes,
                               send_bytes > 0 ? send_bytes : 0,
                               recv_stats.min_nonzero_bytes);
    if (snapshot_request(*request, &info, NULL)) {
      log_request_start(&info, __builtin_return_address(0));
    }
  }
  return rc;
}

int MPI_Ialltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm,
                  MPI_Request *request) {
  int cid = comm_id(comm);
  int csize = comm_size_by_id(cid);
  long long send_one = datatype_bytes(sendtype, sendcount);
  long long recv_one = datatype_bytes(recvtype, recvcount);
  long long send_bytes = send_one * (long long) csize;
  long long recv_bytes = recv_one * (long long) csize;
  RequestInfo info;
  int rc = PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                          recvtype, comm, request);
  if (rc == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL) {
    remember_request(*request, "ialltoall", -1, -1, cid, send_bytes, sendtype);
    set_request_transfer_stats(*request, send_bytes, recv_bytes,
                               send_one > 0 ? csize : 0, recv_one > 0 ? csize : 0,
                               send_one, recv_one,
                               send_one > 0 ? send_one : 0,
                               recv_one > 0 ? recv_one : 0);
    if (snapshot_request(*request, &info, NULL)) {
      log_request_start(&info, __builtin_return_address(0));
    }
  }
  return rc;
}

int MPI_Ialltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                   MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                   const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                   MPI_Request *request) {
  int cid = comm_id(comm);
  int csize = comm_size_by_id(cid);
  CountStats send_stats = count_stats(sendcounts, sendtype, csize);
  CountStats recv_stats = count_stats(recvcounts, recvtype, csize);
  RequestInfo info;
  int rc = PMPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf,
                           recvcounts, rdispls, recvtype, comm, request);
  if (rc == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL) {
    remember_request(*request, "ialltoallv", -1, -1, cid, send_stats.total_bytes, sendtype);
    set_request_transfer_stats(*request, send_stats.total_bytes, recv_stats.total_bytes,
                               send_stats.nonzero, recv_stats.nonzero,
                               send_stats.max_bytes, recv_stats.max_bytes,
                               send_stats.min_nonzero_bytes,
                               recv_stats.min_nonzero_bytes);
    if (snapshot_request(*request, &info, NULL)) {
      log_request_start(&info, __builtin_return_address(0));
    }
  }
  return rc;
}

int MPI_Neighbor_alltoallv(const void *sendbuf, const int sendcounts[],
                           const int sdispls[], MPI_Datatype sendtype,
                           void *recvbuf, const int recvcounts[],
                           const int rdispls[], MPI_Datatype recvtype,
                           MPI_Comm comm) {
  double start = now_us();
  int cid = comm_id(comm);
  int csize = comm_size_by_id(cid);
  CountStats send_stats = count_stats(sendcounts, sendtype, csize);
  CountStats recv_stats = count_stats(recvcounts, recvtype, csize);
  int rc = PMPI_Neighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                                   recvbuf, recvcounts, rdispls, recvtype, comm);
  pthread_mutex_lock(&g_lock);
  log_begin("collective");
  if (g_out) {
    fprintf(g_out,
            ",\"op\":\"neighbor_alltoallv\",\"comm_id\":%d,\"comm_size\":%d,"
            "\"send_bytes\":%lld,\"recv_bytes\":%lld,"
            "\"send_nonzero\":%d,\"recv_nonzero\":%d,"
            "\"send_max_bytes\":%lld,\"recv_max_bytes\":%lld,"
            "\"duration_us\":%.3f,\"buffer_location\":",
            cid, csize, send_stats.total_bytes, recv_stats.total_bytes,
            send_stats.nonzero, recv_stats.nonzero,
            send_stats.max_bytes, recv_stats.max_bytes, now_us() - start);
    json_string(g_out, g_buffer_location);
  }
  log_end(__builtin_return_address(0));
  pthread_mutex_unlock(&g_lock);
  return rc;
}
#endif

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
  RequestInfo info;
  int idx = -1;
  int had = 0;
  double start = now_us();
  MPI_Request original = request ? *request : MPI_REQUEST_NULL;
  if (request) {
    had = snapshot_request(original, &info, &idx);
  }
  int rc = PMPI_Wait(request, status);
  if (rc == MPI_SUCCESS && had) {
    deactivate_request_index(idx);
    log_request_complete(&info, "wait", now_us() - start, __builtin_return_address(0));
  } else if (rc == MPI_SUCCESS && original != MPI_REQUEST_NULL) {
    log_unknown_wait("wait", now_us() - start, __builtin_return_address(0));
  }
  return rc;
}

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) {
  int i;
  RequestInfo *infos = NULL;
  int *indices = NULL;
  int *had = NULL;
  double start = now_us();
  if (count > 0) {
    infos = (RequestInfo *) calloc((size_t) count, sizeof(RequestInfo));
    indices = (int *) calloc((size_t) count, sizeof(int));
    had = (int *) calloc((size_t) count, sizeof(int));
    if (infos && indices && had) {
      for (i = 0; i < count; i++) {
        indices[i] = -1;
        had[i] = snapshot_request(array_of_requests[i], &infos[i], &indices[i]);
      }
    }
  }
  int rc = PMPI_Waitall(count, array_of_requests, array_of_statuses);
  if (rc == MPI_SUCCESS && infos && indices && had) {
    double duration = now_us() - start;
    for (i = 0; i < count; i++) {
      if (had[i]) {
        deactivate_request_index(indices[i]);
        log_request_complete(&infos[i], "waitall", duration, __builtin_return_address(0));
      }
    }
  }
  free(infos);
  free(indices);
  free(had);
  return rc;
}

int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status) {
  int i;
  MPI_Request *before = NULL;
  double start = now_us();
  if (count > 0) {
    before = (MPI_Request *) calloc((size_t) count, sizeof(MPI_Request));
    if (before) {
      for (i = 0; i < count; i++) {
        before[i] = array_of_requests[i];
      }
    }
  }
  int rc = PMPI_Waitany(count, array_of_requests, index, status);
  if (rc == MPI_SUCCESS && index && *index != MPI_UNDEFINED && before) {
    RequestInfo info;
    int idx = -1;
    if (snapshot_request(before[*index], &info, &idx)) {
      deactivate_request_index(idx);
      log_request_complete(&info, "waitany", now_us() - start, __builtin_return_address(0));
    } else {
      log_unknown_wait("waitany", now_us() - start, __builtin_return_address(0));
    }
  }
  free(before);
  return rc;
}

int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {
  RequestInfo info;
  int idx = -1;
  int had = 0;
  double start = now_us();
  MPI_Request original = request ? *request : MPI_REQUEST_NULL;
  if (request) {
    had = snapshot_request(original, &info, &idx);
  }
  int rc = PMPI_Test(request, flag, status);
  if (rc == MPI_SUCCESS && flag && *flag && had) {
    deactivate_request_index(idx);
    log_request_complete(&info, "test", now_us() - start, __builtin_return_address(0));
  }
  return rc;
}

int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag,
                MPI_Status array_of_statuses[]) {
  int i;
  RequestInfo *infos = NULL;
  int *indices = NULL;
  int *had = NULL;
  double start = now_us();
  if (count > 0) {
    infos = (RequestInfo *) calloc((size_t) count, sizeof(RequestInfo));
    indices = (int *) calloc((size_t) count, sizeof(int));
    had = (int *) calloc((size_t) count, sizeof(int));
    if (infos && indices && had) {
      for (i = 0; i < count; i++) {
        indices[i] = -1;
        had[i] = snapshot_request(array_of_requests[i], &infos[i], &indices[i]);
      }
    }
  }
  int rc = PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
  if (rc == MPI_SUCCESS && flag && *flag && infos && indices && had) {
    double duration = now_us() - start;
    for (i = 0; i < count; i++) {
      if (had[i]) {
        deactivate_request_index(indices[i]);
        log_request_complete(&infos[i], "testall", duration, __builtin_return_address(0));
      }
    }
  }
  free(infos);
  free(indices);
  free(had);
  return rc;
}

int MPI_Testany(int count, MPI_Request array_of_requests[], int *index, int *flag,
                MPI_Status *status) {
  int i;
  MPI_Request *before = NULL;
  double start = now_us();
  if (count > 0) {
    before = (MPI_Request *) calloc((size_t) count, sizeof(MPI_Request));
    if (before) {
      for (i = 0; i < count; i++) {
        before[i] = array_of_requests[i];
      }
    }
  }
  int rc = PMPI_Testany(count, array_of_requests, index, flag, status);
  if (rc == MPI_SUCCESS && flag && *flag && index && *index != MPI_UNDEFINED && before) {
    RequestInfo info;
    int idx = -1;
    if (snapshot_request(before[*index], &info, &idx)) {
      deactivate_request_index(idx);
      log_request_complete(&info, "testany", now_us() - start, __builtin_return_address(0));
    }
  }
  free(before);
  return rc;
}
