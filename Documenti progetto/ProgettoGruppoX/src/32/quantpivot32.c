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

    for(int i = 0; i < D; i++){
        vplus[i] = 0;
        vminus[i] = 0;
    }

    /*
    int top_idx[x];      // indici dei top x valori
    float top_val[x];    // valori assoluti corrispondenti
    */

    // versione safe per allocazione vettori

    int *top_idx = malloc(x*sizeof(int));
    float *top_val = malloc(x*sizeof(float));

    int top_count = 0;   // contatore dei top valori trovati finora
    int min_pos = -1;    // indice del minimo tra top x (per sapere quale sostituire quando trovo un nuovo massimo)

    for(int i = 0; i < D; i++){
        float abs_val = fabs(v[i]);
            //se non ho ancora trovato x valori parziali
        if(top_count < x){
            // riempio inizialmente con i primi x valori
            top_idx[top_count] = i;
            top_val[top_count] = abs_val;
            if(min_pos == -1 || abs_val < top_val[min_pos])
                min_pos = top_count;
            top_count++;
        } else if(abs_val > top_val[min_pos]){
            // altrimenti sostituisco il minimo tra i top x trovati finora
            top_idx[min_pos] = i;
            top_val[min_pos] = abs_val;

            // aggiorno anche min_pos per il prossimo giro
            min_pos = 0;
            float min_val = top_val[0];
            for(int j = 1; j < x; j++){
                if(top_val[j] < min_val){
                    min_val = top_val[j];
                    min_pos = j;
                }
            }
        }
    }

    // Infine assegno 1 in vplus o vminus a seconda del segno dei top x valori
    for(int k = 0; k < top_count; k++){
        int idx = top_idx[k];
        if(v[idx] > 0)
            vplus[idx] = 1;
        else if(v[idx] < 0)
            vminus[idx] = 1;
    }
}



int approx_dist(float *w, float *v, int D, int x){ 
    
    // calcola della quantizzazione dei due vettori

    int *vplus = malloc(D*sizeof(int));
    int *vminus = malloc(D*sizeof(int));
    int *wplus = malloc(D*sizeof(int));
    int *wminus = malloc(D*sizeof(int));
    
    quantizzazione(vplus,vminus,v,D,x);
    quantizzazione(wplus,wminus,w,D,x);

    // calcolo della distanza approssimativa tra i due vettori

    int approx_dist = scalar_prod(vplus,wplus,D)+scalar_prod(vminus,wminus,D)-scalar_prod(vplus,wminus,D)-scalar_prod(vminus,wplus,D);

    return approx_dist;

}

int scalar_prod(int *vett1, int *vett2, int D){ //ottimizzabile in assembly facendo operazioni parallele
    
    // calcolo del prodotto scalare tra due vettori

    int s = 0;

    for(int i = 0; i < D; i++){
        s = s + (vett1[i]*vett2[i]);
    }

    return s;

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
