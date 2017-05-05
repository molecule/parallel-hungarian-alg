#include "HungarianAlg.h"
#define NUM_THREADS 256

using namespace std;

int parallel = 1;
int DEBUG = 0;
__device__ double d_columnAnswer;
__device__ double d_rowAnswer;

AssignmentProblemSolver::AssignmentProblemSolver()
{
}

AssignmentProblemSolver::~AssignmentProblemSolver()
{
}

//
//  timer
//
double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {   
        gettimeofday( &start, NULL );
        initialized = true;
    }   
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 ) 
            return i;
    return -1; 
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

double AssignmentProblemSolver::Solve(vector<vector<double> >& DistMatrix,vector<int>& Assignment,TMethod Method)
{
    if(DEBUG) {
        printf("solve\n");
    }
    int N=DistMatrix.size(); // number of columns (tracks)
    int M=DistMatrix[0].size(); // number of rows (measurements)

    int *assignment		=new int[N];
    double *distIn		=new double[N*M];

    double  cost;
    // Fill matrix with random numbers
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            distIn[i+N*j] = DistMatrix[i][j];
        }
    }
    switch(Method)
    {
        case optimal: assignmentoptimal(assignment, &cost, distIn, N, M); break;

        case many_forbidden_assignments: assignmentoptimal(assignment, &cost, distIn, N, M); break;

        case without_forbidden_assignments: assignmentoptimal(assignment, &cost, distIn, N, M); break;
    }

    // form result
    Assignment.clear();
    for(int x=0; x<N; x++)
    {
        Assignment.push_back(assignment[x]);
    }

    delete[] assignment;
    delete[] distIn;
    return cost;
}
// --------------------------------------------------------------------------
// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
// --------------------------------------------------------------------------


__global__ void findMinCol_gpu(double* d_distMatrix, double* d_dualVariablesColumn, int n) {
    int tid = threadIdx.x * blockDim.x;
    if (tid >= n) return;
    int endIndex = tid + blockDim.x;

    d_columnAnswer = d_distMatrix[tid];
    for(int i = tid; i < endIndex; i++) {
        if (d_distMatrix[i] < d_columnAnswer) { d_columnAnswer = d_distMatrix[i]; }	
    }
    //printf("threadIdx.x: %d, tid: %d, endIndex: %d, d_colAnswer: %f\n", threadIdx.x, tid, endIndex, d_columnAnswer);
    d_dualVariablesColumn[threadIdx.x] = d_columnAnswer;

}

__global__ void findMinRow_gpu(double* d_distMatrix, double* d_dualVariablesRow, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    int endIndex = n;

    d_rowAnswer = d_distMatrix[tid];
    for(int i = tid; i < endIndex; i += blockDim.x) {
        if (d_distMatrix[i] < d_rowAnswer) { d_rowAnswer = d_distMatrix[i]; }	
    }
    //printf("tid: %d, endIndex: %d, d_rowAnswer: %f\n", tid, endIndex, d_rowAnswer);
    d_dualVariablesRow[threadIdx.x] = d_rowAnswer;
} 

__global__ void subtractMinElementRow_gpu(double* d_distMatrix, double* d_dualVariablesRow, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    //int endIndex = n; 

    // Subtract the smallest element in this row from each element in this row.
    int nOfRows = sqrt((float)n);
    int rowIdx = threadIdx.x % nOfRows;
    double before = d_distMatrix[tid];
    d_distMatrix[tid] = d_distMatrix[tid] - d_dualVariablesRow[rowIdx];
    printf("subtractMinElemRow, tid: %d, minElem: %f, bef: %f, aft: %f, rowIdx: %d \n", tid, d_dualVariablesRow[rowIdx], before, d_distMatrix[tid], rowIdx);
}

void AssignmentProblemSolver::assignmentoptimal(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns)
{
    if (DEBUG) {
        printf("assignment optimal.\n");
    }
    double *distMatrix;
    double *dualVariablesRow;
    double *dualVariablesColumn;
    double *distMatrixTemp;
    double *distMatrixEnd;
    double *columnEnd;
    double  value;
    double  minValue;

    bool *coveredColumns;
    bool *coveredRows;
    bool *starMatrix;
    bool *newStarMatrix;
    bool *primeMatrix;

    int nOfElements;
    int minDim;
    int row;
    int col;

    // Init
    *cost = 0;
    for(row=0; row<nOfRows; row++)
    {
        assignment[row] = -1.0;
    }

    // Generate distance matrix
    // and check matrix elements positiveness :)

    // Total elements number
    nOfElements   = nOfRows * nOfColumns;
    // Memory allocation
    distMatrix    = (double *)malloc(nOfElements * sizeof(double));
    double * d_distMatrix;
    cudaMalloc((void **) &d_distMatrix, nOfElements * sizeof(double));

    dualVariablesRow = (double *)malloc(nOfRows * sizeof(double));
    double * d_dualVariablesRow;
    cudaMalloc((void **) &d_dualVariablesRow, nOfRows * sizeof(double));
    dualVariablesColumn = (double *)malloc(nOfColumns * sizeof(double));
    double * d_dualVariablesColumn;
    cudaMalloc((void**) &d_dualVariablesColumn, nOfColumns * sizeof(double));

    // Pointer to last element
    distMatrixEnd = distMatrix + nOfElements;

    //
    for(row=0; row<nOfElements; row++)
    {
        value = distMatrixIn[row];
        if(value < 0)
        {
            cout << "All matrix elements have to be non-negative." << endl;
        }
        distMatrix[row] = value;
        //printf("distMatrix[%d]: %f\n", row, value);
    }

    // Memory allocation
    coveredColumns = (bool *)calloc(nOfColumns,  sizeof(bool));
    coveredRows    = (bool *)calloc(nOfRows,     sizeof(bool));
    starMatrix     = (bool *)calloc(nOfElements, sizeof(bool));
    primeMatrix    = (bool *)calloc(nOfElements, sizeof(bool));
    newStarMatrix  = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

    cudaMemcpy(d_distMatrix, distMatrix, nOfElements * sizeof(double), cudaMemcpyHostToDevice);
    //int blks = (nOfElements + NUM_THREADS - 1) / NUM_THREADS;
    //int blks = nOfRows;
    int blks = 1;
    //nOfRows = 1;
    //nOfColumns = 1;
    if (parallel) {
        //findMinCol_gpu <<< blks, nOfRows >>> (d_distMatrix, d_dualVariablesColumn, nOfElements);
        findMinRow_gpu <<< blks, nOfColumns >>> (d_distMatrix, d_dualVariablesRow, nOfElements);
        cudaDeviceSynchronize(); // GPU doesn't block CPU thread
        subtractMinElementRow_gpu <<< blks, nOfElements >>> (d_distMatrix, d_dualVariablesRow, nOfElements);
        cudaDeviceSynchronize(); // GPU doesn't block CPU thread

        //cudaMemcpy(dualVariablesRow, d_dualVariablesRow, nOfRows * sizeof(double), cudaMemcpyDeviceToHost);
        //cudaMemcpy(dualVariablesColumn, d_dualVariablesColumn, nOfColumns * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(distMatrix, d_distMatrix, nOfElements * sizeof(double), cudaMemcpyDeviceToHost);

        if (DEBUG) {
            for(int i = 0; i < nOfElements; i++) {
                printf("distMatrix[%d]: %f\n", i, distMatrix[i]);
            }
        //    for(int i = 0; i < nOfRows; i++) {
        //        printf("smallest value in row %d is: %f\n", i, dualVariablesRow[i]);
        //    }
            /*
            for(int i = 0; i < nOfColumns; i++) {
                printf("smallest value in column %d is: %f\n", i, dualVariablesColumn[i]);
            }
            */
        }
        return;
    } else {
    /* preliminary steps */
        minDim = nOfRows;
        for(row=0; row<nOfRows; row++)
        {
            /* find the smallest element in the row */
            distMatrixTemp = distMatrix + row;
            minValue = *distMatrixTemp;
            distMatrixTemp += nOfRows;
            while(distMatrixTemp < distMatrixEnd)
            {
                value = *distMatrixTemp;
                if(value < minValue)
                {
                    minValue = value;
                }
                distMatrixTemp += nOfRows;
            }
            /* subtract the smallest element from each element of the row */
            distMatrixTemp = distMatrix + row;
            while(distMatrixTemp < distMatrixEnd)
            {
                *distMatrixTemp -= minValue;
                distMatrixTemp += nOfRows;
            }
        }
        /* Steps 1 and 2a */
        for(row=0; row<nOfRows; row++)
        {
            for(col=0; col<nOfColumns; col++)
            {
                if(distMatrix[row + nOfRows*col] == 0)
                {
                    if(!coveredColumns[col])
                    {
                        starMatrix[row + nOfRows*col] = true;
                        coveredColumns[col]           = true;
                        break;
                    }
                }
            }
        }
    }
    /* move to step 2b */
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    /* compute cost and remove invalid assignments */
    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
    /* free allocated memory */
    free(distMatrix);
    free(coveredColumns);
    free(coveredRows);
    free(starMatrix);
    free(primeMatrix);
    free(newStarMatrix);
    return;
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns)
{
    if (DEBUG) {printf("build assignment vector.\n");}
    int row, col;
    for(row=0; row<nOfRows; row++)
    {
        for(col=0; col<nOfColumns; col++)
        {
            if(starMatrix[row + nOfRows*col])
            {
                assignment[row] = col;
                break;
            }
        }
    }
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows)
{
    if (DEBUG) { printf("compute assignment cost.\n");}
    int row, col;
    for(row=0; row<nOfRows; row++)
    {
        col = assignment[row];
        if(col >= 0)
        {
            *cost += distMatrix[row + nOfRows*col];
        }
    }
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    if (DEBUG) { printf("step 2a\n"); }
    bool *starMatrixTemp, *columnEnd;
    int col;
    /* cover every column containing a starred zero */
    for(col=0; col<nOfColumns; col++)
    {
        starMatrixTemp = starMatrix     + nOfRows*col;
        columnEnd      = starMatrixTemp + nOfRows;
        while(starMatrixTemp < columnEnd)
        {
            if(*starMatrixTemp++)
            {
                coveredColumns[col] = true;
                break;
            }
        }
    }
    /* move to step 3 */
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    if (DEBUG) { printf("step 2b\n");}
    int col, nOfCoveredColumns;
    /* count covered columns */
    nOfCoveredColumns = 0;
    for(col=0; col<nOfColumns; col++)
    {
        if(coveredColumns[col])
        {
            nOfCoveredColumns++;
        }
    }
    if(nOfCoveredColumns == minDim)
    {
        /* algorithm finished */
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    }
    else
    {
        /* move to step 3 */
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    if (DEBUG) {printf("step 3\n");}
    bool zerosFound;
    int row, col, starCol;
    zerosFound = true;
    while(zerosFound)
    {
        zerosFound = false;
        for(col=0; col<nOfColumns; col++)
        {
            if(!coveredColumns[col])
            {
                for(row=0; row<nOfRows; row++)
                {
                    if((!coveredRows[row]) && (distMatrix[row + nOfRows*col] == 0))
                    {
                        /* prime zero */
                        primeMatrix[row + nOfRows*col] = true;
                        /* find starred zero in current row */
                        for(starCol=0; starCol<nOfColumns; starCol++)
                            if(starMatrix[row + nOfRows*starCol])
                            {
                                break;
                            }
                        if(starCol == nOfColumns) /* no starred zero found */
                        {
                            /* move to step 4 */
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                        else
                        {
                            coveredRows[row]        = true;
                            coveredColumns[starCol] = false;
                            zerosFound              = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    /* move to step 5 */
    step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{
    if (DEBUG) { printf("step 4\n");}
    int n, starRow, starCol, primeRow, primeCol;
    int nOfElements = nOfRows*nOfColumns;
    /* generate temporary copy of starMatrix */
    for(n=0; n<nOfElements; n++)
    {
        newStarMatrix[n] = starMatrix[n];
    }
    /* star current zero */
    newStarMatrix[row + nOfRows*col] = true;
    /* find starred zero in current column */
    starCol = col;
    for(starRow=0; starRow<nOfRows; starRow++)
    {
        if(starMatrix[starRow + nOfRows*starCol])
        {
            break;
        }
    }
    while(starRow<nOfRows)
    {
        /* unstar the starred zero */
        newStarMatrix[starRow + nOfRows*starCol] = false;
        /* find primed zero in current row */
        primeRow = starRow;
        for(primeCol=0; primeCol<nOfColumns; primeCol++)
        {
            if(primeMatrix[primeRow + nOfRows*primeCol])
            {
                break;
            }
        }
        /* star the primed zero */
        newStarMatrix[primeRow + nOfRows*primeCol] = true;
        /* find starred zero in current column */
        starCol = primeCol;
        for(starRow=0; starRow<nOfRows; starRow++)
        {
            if(starMatrix[starRow + nOfRows*starCol])
            {
                break;
            }
        }
    }
    /* use temporary copy as new starMatrix */
    /* delete all primes, uncover all rows */
    for(n=0; n<nOfElements; n++)
    {
        primeMatrix[n] = false;
        starMatrix[n]  = newStarMatrix[n];
    }
    for(n=0; n<nOfRows; n++)
    {
        coveredRows[n] = false;
    }
    /* move to step 2a */
    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    if (DEBUG) { printf("step 5\n");}
    double h, value;
    int row, col;
    /* find smallest uncovered element h */
    h = DBL_MAX;
    for(row=0; row<nOfRows; row++)
    {
        if(!coveredRows[row])
        {
            for(col=0; col<nOfColumns; col++)
            {
                if(!coveredColumns[col])
                {
                    value = distMatrix[row + nOfRows*col];
                    if(value < h)
                    {
                        h = value;
                    }
                }
            }
        }
    }
    /* add h to each covered row */
    for(row=0; row<nOfRows; row++)
    {
        if(coveredRows[row])
        {
            for(col=0; col<nOfColumns; col++)
            {
                distMatrix[row + nOfRows*col] += h;
            }
        }
    }
    /* subtract h from each uncovered column */
    for(col=0; col<nOfColumns; col++)
    {
        if(!coveredColumns[col])
        {
            for(row=0; row<nOfRows; row++)
            {
                distMatrix[row + nOfRows*col] -= h;
            }
        }
    }
    /* move to step 3 */
    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}


// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases without forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal2(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns)
{
    printf("assignmentsuboptimal2...............................................\n");
}

// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases with many forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal1(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns)
{
    printf("assignmentsuboptimal1..............................................\n");
}
// --------------------------------------------------------------------------
// Usage example
// --------------------------------------------------------------------------
//int main(void)
int main( int argc, char **argv )
{
    int n = read_int( argc, argv, "-n", 10 );
    int m = read_int( argc, argv, "-m", 10);
    int print = read_int( argc, argv, "-p", 0);
    parallel = read_int( argc, argv, "-L", 1);
    DEBUG = read_int( argc, argv, "-d", 0);
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    // Matrix size
    int N=n; // tracks rows
    int M=m; // detects columns
    // Random numbers generator initialization
    srand (time(NULL));
    // Distance matrix N-th track to M-th detect.
    vector< vector<double> > Cost(N,vector<double>(M));
    // Fill matrix with random values
    printf("HungarianAlg.cpp\n");
    printf("Creating a random Cost Matrix:\n");
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            Cost[i][j] = (double)(rand()%1000)/1000.0;
            if (print) { std::cout << Cost[i][j] << "\t";}
        }
        if (print) { std::cout << std::endl;}
    }
    AssignmentProblemSolver APS;
    vector<int> Assignment;
    printf("Solving the random matrix...\n");
    double solve_time = read_timer( );
    double totalCost = APS.Solve(Cost, Assignment);
    //cout << APS.Solve(Cost,Assignment) << endl;
    solve_time = read_timer( ) - solve_time;
    printf("Total solve_time: %g\n", solve_time);

    // Output the result
    if (print) {
        for(int x=0; x<N; x++)
        {
            std::cout << x << ":" << Assignment[x] << "\t";
        }
    }
}
// --------------------------------------------------------------------------
