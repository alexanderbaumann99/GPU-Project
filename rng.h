/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/
#include <iostream>
#include <math.h>
#include <stdlib.h>


using namespace std;

#define Mtraj (1048576)
#define MoPI (3.1415927f)

////////////////////////////////////////////////////////////////
// L'Eucuyer CMRG Matrix Values
////////////////////////////////////////////////////////////////
// First MRG 
#define a12 63308
#define a13 -183326
#define q12 33921
#define q13 11714
#define r12 12979
#define r13 2883

// Second MRG 
#define a21 86098
#define a23 -539608
#define q21 24919
#define q23 3976
#define r21 7417
#define r23 2071

// Normalization variables
#define Invmp   4.6566129e-10f                  
#define two17   131072.0
#define two53   9007199254740992.0

////////////////////////////////////////////////////////////////
// Datatype definitions
////////////////////////////////////////////////////////////////
// CMRG datatype
typedef int TabSeedCMRG_t[Mtraj][6];

////////////////////////////////////////////////////////////////
// Memory for RNG use 
////////////////////////////////////////////////////////////////
// The state variables of CMRG on GPU 
extern TabSeedCMRG_t *CMRG;
// The state variables of CMRG on CPU
extern TabSeedCMRG_t *CMRGp;
// Matrixes associated to the post treatment of the CMRG
// - First MRG
extern double A1[3][3];
// - Second MRG
extern double A2[3][3];

////////////////////////////////////////////////////////////////
// Post initialization of CMRG
////////////////////////////////////////////////////////////////
void PostInitDataCMRG(void);

void FreeCMRG(void);