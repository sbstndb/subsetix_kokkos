#pragma once

/**
 * @file mpi_stub.hpp
 *
 * @brief Stub MPI pour compilation sans MPI
 *
 * Ce fichier fournit des définitions stub pour MPI lorsqu'il n'est pas disponible.
 * Permet à l'API de compiler en mode single-rank sans MPI installé.
 */

#ifdef SUBSETIX_USE_MPI
#include <mpi.h>
#else

// Stub types et constantes pour compilation sans MPI

typedef int MPI_Comm;
typedef int MPI_Request;
typedef int MPI_Op;
typedef int MPI_Datatype;
typedef int MPI_Status;

// Constantes
#define MPI_COMM_WORLD 0
#define MPI_SUM 0
#define MPI_MAX 1
#define MPI_MIN 2
#define MPI_PROD 3
#define MPI_LAND 4
#define MPI_LOR 5
#define MPI_BXOR 6
#define MPI_BOR 7
#define MPI_LXOR 8
#define MPI_BAND 9

#define MPI_FLOAT 0
#define MPI_DOUBLE 1
#define MPI_INT 2
#define MPI_LONG 3
#define MPI_UNSIGNED 4
#define MPI_UNSIGNED_LONG 5

#define MPI_MAX_PROCESSOR_NAME 256
#define MPI_ANY_SOURCE -1
#define MPI_ANY_TAG -1
#define MPI_PROC_NULL -2

#define MPI_SUCCESS 0
#define MPI_ERR_COMM 1
#define MPI_ERR_TAG 2
#define MPI_ERR_COUNT 3

// Metis stub
#define METIS_NOPTIONS 40

// Stub functions (inline, no-op)
inline int MPI_Init(int* argc, char*** argv) { return 0; }
inline int MPI_Finalize() { return 0; }
inline int MPI_Comm_rank(MPI_Comm, int* rank) { *rank = 0; return 0; }
inline int MPI_Comm_size(MPI_Comm, int* size) { *size = 1; return 0; }
inline int MPI_Barrier(MPI_Comm) { return 0; }
inline int MPI_Bcast(void*, int, MPI_Datatype, int, MPI_Comm) { return 0; }
inline int MPI_Send(const void*, int, MPI_Datatype, int, int, MPI_Comm) { return 0; }
inline int MPI_Recv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status*) { return 0; }
inline int MPI_Isend(const void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*) { return 0; }
inline int MPI_Irecv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*) { return 0; }
inline int MPI_Wait(MPI_Request*, MPI_Status*) { return 0; }
inline int MPI_Waitall(int, MPI_Request[], MPI_Status[]) { return 0; }
inline int MPI_Allreduce(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm) { return 0; }

#endif // SUBSETIX_USE_MPI
