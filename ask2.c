#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// struct for the index and the contents of this index (excersize d)
struct d
{
    int index;
    float number;
};

// initialization of all the functions used in this program
void UserInput(int **, int *, int);

void calculateVectors(int, int, int, int **, int **);

void distributeToProcessor(int *, int *, int *, int **, int *, int *, int *, int, int, float **);

void exc_a(int, int, float *, int *, int *, int, int, int *);

void exc_b(int, int, int, int, int *, float, float *);

void exc_c(int, int, int, int, int, float **, float *, int *, int, int *, int *);

void exc_d(int, float *, int, struct d *, int);

void exc_e(int *, int **, int, int, int **, int);

//menu function
int menu(int *, int);

/*
    The UserInput function is the first function that runs in the program.
    In here the inputs of the user are used for the vector's size(size) 
    and the vector's contents. The vector X used in this program is being
    initiallized here and it's numbers are being printed out as an output. 
*/
void UserInput(int **X, int *N, int size)
{

    int size_of_vector;

    //user input for the vector's size
    printf("Please enter the vector's size.\n");
    scanf("%d", &size_of_vector);

    *N = size_of_vector;

    // initiates the table and allocates the memory dynamically
    (*X) = (int *)malloc(size_of_vector * sizeof(int));

    // checks if the vector is empty and if it is, then the program returns error and exits the MPI
    if (*X == NULL)
    {
        printf("Error! Memory allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("Give the vector's numbers:\n");

    // asks the user for input to fill up the array
    for (int i = 0; i < size_of_vector; i++)
    {
        printf("Enter the numbers. \n");
        printf("X[%d]: ", i);
        scanf("%d", &((*X)[i]));
    }
}

/*
    In the calculateVectors function the tables local_size and global_index are being
    allocated for later use in the program. The local_size table contains the local 
    sizes of every sub-vector that each processor has, of the greater vector X. This
    function makes it possible for the vector X to be divided evenly accross all 
    different processors.
*/
void calculateVectors(int size, int rank, int N, int **global_index, int **local_size)
{
    int div = N / size;
    int remainder = N % size;

    // Allocate memory for local_size, local_size is a table that holds all local_sizes of every
    // table that gets distributed to a processor
    *local_size = (int *)malloc(size * sizeof(int));
    if (*local_size == NULL)
    {
        printf("Rank %d: Memory allocation failed (local_size).\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate memory for global_index, global_index is a table that holds all indexes of table X
    *global_index = (int *)malloc(size * sizeof(int));
    if (*global_index == NULL)
    {
        printf("Rank %d: Memory allocation failed (global_index).\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Calculate local_size and global_index
    int offset = 0;
    for (int i = 0; i < size; i++)
    {
        if (remainder > 0)
        {
            (*local_size)[i] = div + 1;
            remainder--;
        }
        else
        {
            (*local_size)[i] = div;
        }
        (*global_index)[i] = offset;
        offset += (*local_size)[i];
    }
}

/*
    In the distributeToProcessor function values like local_size, global_index and 
    local_x are being either broadcasted or scattered accross the program. In this 
    function the sub-vectors of each processor for both local_x and local_D are 
    being allocated, and later with the use of MPI_Scatterv, the data are being 
    distributed.
*/
void distributeToProcessor(int *local_size, int *global_index, int *local_n, int **local_x, int *N, int *first_index, int *X, int rank, int size, float **local_D)
{
    MPI_Bcast(N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Broadcasted N = %d\n", *N);

    // Broadcast global_index and local_size to all processors
    if (rank != 0)
    {
        local_size = (int *)malloc(size * sizeof(int));
        global_index = (int *)malloc(size * sizeof(int));
    }
    MPI_Bcast(local_size, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_index, size, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter sizes and indices
    MPI_Scatter(local_size, 1, MPI_INT, local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(global_index, 1, MPI_INT, first_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // printf("Rank %d: Received first_index = %d, local_n = %d\n", rank, *first_index, *local_n);

    // Allocate memory for local_x
    *local_x = (int *)malloc((*local_n) * sizeof(int));
    if (*local_x == NULL)
    {
        printf("Rank %d: Memory allocation failed (local_x).\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    *local_D = (float *)malloc((*local_n) * sizeof(float));
    if (*local_D == NULL)
    {
        printf("Rank %d: Memory allocation failed (local_D).\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Scatter vector data
    MPI_Scatterv(X, local_size, global_index, MPI_INT, *local_x, *local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Print received data
    // printf("Rank %d: Received local_x = ", rank);
    // for (int i = 0; i < *local_n; i++) {
    //     printf("%d ", (*local_x)[i]);
    // }
    // printf("\n");
}

/*
    This function solves the excersize a. Firstly, it calculates the local_sum for each 
    processor and then finds the max and min value for each sub-vector. Then the data 
    are sent to processor 0 and with MPI_Reduce it calculates the average, min and max
    for the greater vector X. Then these values are being broadcasted through all the
    processors. 
*/
void exc_a(int rank, int size, float *average, int *min, int *max, int N, int local_n, int *local_x)
{

    float local_sum = 0;
    int local_max = local_x[0];
    int local_min = local_x[0];

    //for loop to calculate local_sum, min and max for each sub-vector 
    for (int i = 0; i < local_n; i++)
    {
        local_sum += local_x[i];
        if (local_max < local_x[i])
        {
            local_max = local_x[i];
        }

        if (local_min > local_x[i])
        {
            local_min = local_x[i];
        }
    }

    // debug printf statement
    //  printf("Rank %d: Local sum = %.2f, Local min = %d, Local max = %d\n", rank, local_sum, local_min, local_max);

    local_sum = local_sum / N;

    //processor 0 gathers valeues local_min, local_max and local_sum and with MPI_Reduce
    //calculates the values for the vector X
    MPI_Reduce(&local_sum, average, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // another debug printf statement for the values that processor 0 (root) receives
    //  if (rank == 0) {
    //      printf("Global results on Rank 0: Average = %.2f, Min = %d, Max = %d\n", *average, *min, *max);
    //  }

    //Processor 0 broadcasts the data to all processors
    MPI_Bcast(average, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(min, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(max, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // printf("Rank %d: Final results - Average = %.2f, Min = %d, Max = %d\n", rank, *average, *min, *max);
}

/*
    This function solves excersize b. It calculates individual var for 
    each processor according to it's sub-vector. Then the result with 
    MPI_Reduce is being sent to processor 0 which then calculates the 
    final_var value after each processor's result. 
*/
void exc_b(int rank, int size, int local_n, int N, int *local_x, float average, float *var)
{
    float indiv_var = 0;

    //for loop for calculation of var for each processor's sub-vector 
    for (int i = 0; i < local_n; i++)
    {
        indiv_var += (float)(local_x[i] - average) * (float)(local_x[i] - average);
    }

    // debug printf statement of individual contributions to variance
    //  printf("Rank %d: Local variance contribution (before division by N) = %.2lf\n", rank, indiv_var);

    // calculates each processor's individual var
    //  indiv_var/=N;

    // processor 0 receives all individual var from every processor
    MPI_Reduce(&indiv_var, var, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 debug printf statement the final variance
    //  if (rank == 0) {
    //      printf("Rank %d: Global variance = %.2lf\n", rank, *var/N);
    //  }
}

/*
    This function solves excersize c. Firstly, the min/max values are being checked.
    They can not be equal due to the division happening later in a calculation. Then 
    table D is being allocated properly and the value for local_D sub-vector, for 
    each processor, is being calculated. At last the values are sent with MPI_Gatherv
    to processor 0. 
*/
void exc_c(int rank, int size, int min, int max, int N, float **D, float *local_D, int *local_x, int local_n, int *global_index, int *local_size)
{
    // Check if max equals min
    if (max == min)
    {
        if (rank == 0)
            printf("Error: Max and Min are equal, cannot normalize.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocation of table D on rank 0
    if (rank == 0)
    {

        // debug printf statement for each processor when allocating table D
        //  printf("Rank %d: Allocating memory for D of size %d.\n", rank, N);

        // allocation of table D
        (*D) = (float *)malloc(N * sizeof(float));

        // error handling for the allocation
        if (*D == NULL)
        {
            printf("Rank %d: Memory allocation failed (D).\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // debug printf statement for local_x data
    //  printf("Rank %d: Local data (before normalization): ", rank);
    //  for (int i = 0; i < local_n; i++) {
    //      printf("%d ", local_x[i]);
    //  }
    //  printf("\n");

    //for loop for local_D calculation 
    for (int i = 0; i < local_n; i++)
    {
        local_D[i] = ((float)(local_x[i] - min) / (max - min)) * 100;
    }

    // debug printf statement for local_D data
    //  printf("Rank %d: Local data (after normalization): ", rank);
    //  for (int i = 0; i < local_n; i++) {
    //      printf("%.2f ", local_D[i]);
    //  }
    //  printf("\n");

    // debug printf statements for Gatherv input parameters
    //  if (rank == 0) {
    //      printf("Rank %d: Gatherv parameters:\n", rank);
    //      printf("  global_index (displs): ");
    //      for (int i = 0; i < size; i++) {
    //          printf("%d ", global_index[i]);
    //      }
    //      printf("\n");

    //     printf("  local_size (recvcounts): ");
    //     for (int i = 0; i < size; i++) {
    //         printf("%d ", local_size[i]);
    //     }
    //     printf("\n");
    // }

    // gather normalized data
    MPI_Gatherv(local_D, local_n, MPI_FLOAT, *D, local_size, global_index, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // debug printf statement for gathered data on rank 0
    //  if (rank == 0) {
    //      printf("Rank %d: Gathered data in D: ", rank);
    //      for (int i = 0; i < N; i++) {
    //          printf("%.1f ", (*D)[i]);
    //      }
    //      printf("\n");
    // }
}

/*
    This function solves excersize d. This function uses a struct d that contains values number 
    and index. These values are later used to calculate the position of the max value 
    first in the sub-vectors (local_D) of each processor and the with MPI_Reduce, for the greater
    D vector. 
*/
void exc_d(int first_index, float *local_D, int local_n, struct d *max_position, int rank)
{

    struct d local_max_position;

    // sets the number and index variable on the first cell of table local_D and on the first index
    // in order to start from the beggining of the table.
    local_max_position.number = local_D[0];
    local_max_position.index = first_index + 0;

    // for loop to calculate the max number of each processor's local_D table and set the correct index
    for (int i = 0; i < local_n; i++)
    {
        if (local_max_position.number < local_D[i])
        {
            local_max_position.number = local_D[i];
            local_max_position.index = first_index + i;
        }
    }

    // debug printf statement to check for the values after the for loop
    //  printf("Processor %d: Final local_max_position.number = %.1f, local_max_position.index = %d \n", rank, local_max_position.number, local_max_position.index);

    // processor 0 uses MPI_Reduce with data type MPI_FLOAT_INT passing data from struct
    // then with the use of MPI_MAXLOC calculates the location of the max and it's index
    MPI_Reduce(&local_max_position, max_position, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
}

/*
    This function solves excersize e. This function calculates the prefix Sums of vector X.
    Firstly, globalprefixSum table and localprefixSum table are being allocated. Then the 
    localprefixSum table is being filled up with data according to the algorithm for prefix
    sum calculation. Then according to the sum of the local data (local_sum), MPI_Scan is 
    used to calculate the offset that is later added in the local prefix. MPI_Scan calculates
    the prefix sums for the processor's sub-vector each time. At last the offset is being 
    calculated as the substruction of scan_results with the last index of the local prefix. 
    That way every time the offset is being added, for the shift to be done properly. The 
    localprefisSum table is being sent to processor 0 and is later being saved in the 
    globalprefixSum table.  
*/
void exc_e(int *local_x, int **globalprefixSum, int rank, int N, int **localprefixSum, int local_n)
{
    if (rank == 0)
    {   
        //globalprefixSum table allocation
        (*globalprefixSum) = (int *)malloc(N * sizeof(int));

        if (*globalprefixSum == NULL)
        {
            printf("Rank %d: Memory allocation failed (globalprefixSum).\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    //localprefixSum table allocation
    *localprefixSum = (int *)malloc((local_n) * sizeof(int));
    if (*localprefixSum == NULL)
    {
        printf("Rank %d: Memory allocation failed (localprefixSum).\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // fill the localprefixSum table
    for (int i = 0; i < local_n; i++)
    {
        if (i == 0)
        {
            (*localprefixSum)[i] = local_x[i];
        }
        else
        {
            (*localprefixSum)[i] = (*localprefixSum)[i - 1] + local_x[i];
        }
    }

    // printf("Rank %d: localprefixSum after computation: ", rank);
    // for (int i = 0; i < local_n; i++) {
    //     printf("%d ", (*localprefixSum)[i]);
    // }
    // printf("\n");

    // calculation of globalprefixSum with MPI_Scan
    int local_sum = (*localprefixSum)[local_n - 1]; // the last value of localprefixSum table
    int scan_results;
    MPI_Scan(&local_sum, &scan_results, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // printf("Rank %d: local_sum after MPI_Scan: %d\n", rank, local_sum);

    // adjust the localprefixSum with the offset the MPI_Scan returned
    int offset = scan_results - (*localprefixSum)[local_n - 1];
    // printf("Rank %d: Offset after MPI_Scan: %d\n", rank, offset);
    for (int i = 0; i < local_n; i++)
    {
        (*localprefixSum)[i] += offset;
    }

    // printf("Rank %d: localprefixSum after offset adjustment: ", rank);
    // for (int i = 0; i < local_n; i++) {
    //     printf("%d ", (*localprefixSum)[i]);
    // }
    // printf("\n");

    // gather all the data in globalprefixSum table
    MPI_Gather(*localprefixSum, local_n, MPI_INT, *globalprefixSum, local_n, MPI_INT, 0, MPI_COMM_WORLD);
}

/*
    This is a menu function. It uses a do...while loop and asks for user input in 
    order to loop the program. If the user's input is between 1 and 2 the while 
    loop exits and then the answer is being broadcasted to the program. If the 
    answer is 1 the program runs again, if the answer is 2 the program exits
    and MPI_Finilize runs. 
*/
int menu(int *option, int size)
{

    int input;

    printf("------------------MENU------------------\n");
    printf("To conitnue press: 1\n");
    printf("To exit press: 2\n");

    do
    {
        scanf("%d", &input);

        // checks if the input is valid. Returns error if it is not
        // and asks for the users' input again, until it is valid.
        if (input > 2 || input < 1)
        {
            printf("_INVALID_INPUT_! Please enter a number between 1-2.\n");
        }
    } while (input > 2 || input < 1);

    *option = input;
}

int main(int argc, char *argv[])
{
    int rc, rank, size;
    int N;
    int *X = NULL, *global_index = NULL, *local_size = NULL;
    float *D = NULL;
    float *local_D;
    int local_n, *local_x = NULL;
    int first_index;

    int maxX, minX;
    float average;
    float var;
    int global_sum;

    struct d max_position;

    int *localprefixSum;
    int *globalprefixSum;
    int option;

    // Initialization of MPI
    rc = MPI_Init(&argc, &argv);

    // checks for error in the initialization
    if (rc != 0)
    {
        printf("MPI initialization error\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // initializes the status variable
    MPI_Status status;

    // initalizes the rank for every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // initializes the amount of processors this program uses
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    do
    {
        if (rank == 0)
        {   
            //those two functions run in the processor 0
            UserInput(&X, &N, size);
            calculateVectors(size, rank, N, &global_index, &local_size);
        }

        //calling of the functions 
        distributeToProcessor(local_size, global_index, &local_n, &local_x, &N, &first_index, X, rank, size, &local_D);

        exc_a(rank, size, &average, &minX, &maxX, N, local_n, local_x);

        exc_b(rank, size, local_n, N, local_x, average, &var);

        exc_c(rank, size, minX, maxX, N, &D, local_D, local_x, local_n, global_index, local_size);

        exc_d(first_index, local_D, local_n, &max_position, rank);

        exc_e(local_x, &globalprefixSum, rank, N, &localprefixSum, local_n);

        // Print results in the main function
        if (rank == 0)
        {
            printf("-----------------excersize a-----------------\n");
            printf("Results verified in main on Rank 0:\n");
            printf("Average = %.2lf, Min = %d, Max = %d\n", average, minX, maxX);

            printf("-----------------excersize b-----------------\n");
            printf("global var is: %.2lf\n", var / N);

            printf("-----------------excersize c-----------------\n");

            for (int i = 0; i < N; i++)
            {
                printf("D[%d] = %.1f \n", i, D[i]);
            }

            printf("-----------------excersize d-----------------\n");
            printf("Processor %d: Global max found - max_position.number = %.1f, max_position.index = %d\n", rank, max_position.number, max_position.index);

            printf("-----------------excersize e-----------------\n");
            printf("Rank %d: globalprefixSum after MPI_Gather: ", rank);
            for (int i = 0; i < N; i++)
            {
                printf("%d ", globalprefixSum[i]);
            }
            printf("\n");

            menu(&option, size);
        }
            MPI_Bcast(&option, 1, MPI_INT, 0, MPI_COMM_WORLD);


        // Free allocated memory
        if (rank == 0)
        {
            free(X);
            free(D);
            free(globalprefixSum);
        }

        free(global_index);
        free(local_size);
        free(local_x);
        free(local_D);
        free(localprefixSum);
    } while (option == 1);


    MPI_Finalize();
    return 0;
}