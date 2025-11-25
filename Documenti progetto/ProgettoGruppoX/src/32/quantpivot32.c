#include <stdlib.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <math.h>
#include <omp.h>
#include "common.h"

extern void prova(params* input);

void quantizzazione(int *vplus, int *vminus, float *v, int D, int x){

    if(D <= 0){
        fprintf(stderr, "Errore nel parametro dimensione D\n");
        exit(1);
    }

    if(x >= D || x <= 0){
        fprintf(stderr, "Errore nel parametro di quantizzazione\n");
        exit(1);
    }

    for(int i = 0; i < size(v); i++){    
        vplus[i] = 0;
        vminus[i] = 0;
    }

    // trovare valori assoluti in v

    // trovare gli x valori assoluti maggiori

    // assegnare 1 agli indici dei valori maggiori positivi in vplus 

    // assegnare 1 agli indici dei valori maggiori negativi in vminus

}

void fit(params* input){
    // Selezione dei pivot
    // Costruzione dell'indice
    input->index = _mm_malloc(8*sizeof(type), align);




}

void predict(params* input){
    // Esecuzione delle query
    input->id_nn[1] = 5;
    prova(input);
}
