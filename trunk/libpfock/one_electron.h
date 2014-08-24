#ifndef __ONE_ELECTRON_H__
#define __ONE_ELECTRON_H__


#include "CInt.h"
#include "pfock_def.h"


void compute_S (PFock_t pfock, BasisSet_t basis,
                int startshellrow, int endshellrow,
                int startshellcol, int endshellcol,
                double *S, int ldS);

void compute_H (PFock_t pfock, BasisSet_t basis,
                int startshellrow, int endshellrow,
                int startshellcol, int endshellcol,
                double *H, int ldH);


#endif /* __ONE_ELECTRON_H__ */
