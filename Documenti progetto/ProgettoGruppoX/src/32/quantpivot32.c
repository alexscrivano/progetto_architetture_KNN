#include <stdlib.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <math.h>
#include <omp.h>
#include "common.h"

extern void prova(params* input);

void quantizzazione(int *vplus, int *vminus, type *v, int D, int x){

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
    type *top_val = malloc(x*sizeof(type));

    int top_count = 0;   // contatore dei top valori trovati finora
    int min_pos = -1;    // indice del minimo tra top x (per sapere quale sostituire quando trovo un nuovo massimo)

    for(int i = 0; i < D; i++){
        type abs_val = fabs(v[i]);
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
            type min_val = top_val[0];
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
        if(v[idx] >= 0)
            vplus[idx] = 1;
        else if(v[idx] < 0)
            vminus[idx] = 1;
    }

    free(top_idx);
    free(top_val);
}

type approx_dist(type *w, type *v, int D, int x){ 
    
    // calcola della quantizzazione dei due vettori

    int *vplus = malloc(D*sizeof(int));
    int *vminus = malloc(D*sizeof(int));
    int *wplus = malloc(D*sizeof(int));
    int *wminus = malloc(D*sizeof(int));

    if (!vplus || !vminus || !wplus || !wminus) {
        fprintf(stderr, "Errore malloc in approx_dist\n");
        exit(1);
    }
    
    quantizzazione(vplus,vminus,v,D,x);
    quantizzazione(wplus,wminus,w,D,x);

    // calcolo della distanza approssimativa tra i due vettori

    type approx_dist = scalar_prod(vplus,wplus,D)+scalar_prod(vminus,wminus,D)-scalar_prod(vplus,wminus,D)-scalar_prod(vminus,wplus,D);

    free(vplus);
    free(vminus);
    free(wplus);
    free(wminus);

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

void selezione_pivot(params* input){

    int h = input->h;
    int N = input->N;
    
    if(h <= 0 || h >= N){
        fprintf(stderr,"Errore nel numero di pivot h\n");
        exit(1);
    }

    // selezione dei punti pivot casualmente

    int index = 0;

    int step = N/h;

    while (index < h){
        int i = step * index;
        //insiemePivot[index] = dset[i];
        input->P[index] = i;//input->DS[i]; //input->P + (size_t)p*D;
        index ++;
    }
    
    if (!input->silent) {
        printf("Selezionati %d pivot (step=%d)\n", h, step);
    }

}

void indexing(params* input){ // ottimizzabile con la questione della vettorizzazione della matrice

    // indicizzazione delle distanze approssimate dei punti del dataset da ciascun pivot

    int P = input->h;
    int N = input->N;
    int D = input->D;
    int x = input->x;

    for(int p = 0; p < P; p++){
        int idx_pivot = input->P[p];                   // indice nel DS
        type *pivot   = input->DS + (size_t)idx_pivot * D;
        type *row = input->index + (size_t)p*N;
        for(int i = 0; i < N; i++){

            type *dset_row = input->DS + (size_t)i *D;

            row[i] = approx_dist(pivot,dset_row,D,x);
        }
    }

    if (!input->silent) {
        printf("Costruito indice pivot (%d pivot x %d punti)\n", P, N);
    }

}

void fit(params* input){
    // Selezione dei pivot
    // Costruzione dell'indice
    
    if(!input->DS || input->N <= 0 || input->D <= 0){
        fprintf(stderr, "DS non inizializzato correttamente\n");
        exit(1);
    }
    if (!input->P || input->h <= 0) {
        fprintf(stderr, "Vettore P (pivot) non allocato o h non valido\n");
        exit(1);
    }
    
    size_t index_size = (size_t)input->h * input->N * sizeof(type);
    input->index = _mm_malloc(index_size, align);

    if (!input->index) {
        fprintf(stderr, "Errore allocazione index (%zu bytes)\n", index_size);
        exit(1);
    }

    if (!input->silent) {
        printf("Inizio funzione FIT: selezione pivot + costruzione indice\n");
    }

    selezione_pivot(input);
    indexing(input);

    if (!input->silent) {
        printf("FIT completato\n");
    }
}

/*
void querying(params* input, type* q){ // ottimizzabile con la vettorizzazione delle matrici in assembly
    
    // inizializzazione dei KNN di q
    int k = input->k;
    int nq = input->nq;
    int N = input->N;

    for(int i = 0; i < k; i++){
        input->id_nn[i] = -1;
    }

    for(int i = 0; i < nq*k; i++){
        input->dist_nn[i] = INFINITY;
    }

    int P = input->h;
    int D = input->D;

    // calcolo delle distanze approssimate di ciascun pivot dal punto di query q

    type* distanze_app = malloc(P*sizeof(type));

    for(int i = 0; i < P; i++){
        type *pivot = input->P + (size_t)i*D;
        distanze_app[i] = approx_dist(q,pivot,D,input->x);
    }

    int N = input->N;
    type* index = input->index;

    for(int v = 0; v < N; v++){
        type* row = input->DS + (size_t)v*D;
        type dpvt = max_distance(distanze_app, index); //ottimizzabile con calcolo in parallelo su assembly
        type dmax = calcola(); 
        if(dpvt<dmax){
            type dist = approx_dist(row,q,D,input->x); 
            if(dist < dmax){
                aggiornaKNN(input->dist_nn,input->id_nn); 
            }
        }    
    }

    for(int i = 0; i < k; i++){
        int* id = input->id_nn + (size_t)i;
        type* vid = input->DS + (size_t)id * D;
        
        type delta = calcola_distanza(q,vid);
        input->dist_nn[i*nq] = delta;
    }

    free(distanze_app);

}

*/



type distanza(type *v, type *w, int D){ // ottimizzabile in assembly
    type distanza = 0;
    // calcolo delle distanze al quadrato
    for(int i = 0; i < D; i++){
        distanza += pow((v[i] - w[i]),2);
    }
    // calcolo della radice della distanza
    distanza = sqrt(distanza);
    return distanza;
}

void insert_nn(int *id_nn_q, type *dist_nn_q, int v, type dist, int k){
    // inserimento del vicino di indice v e della distanza approssimata di v da q

    int indice_max = 0;
    type max = dist_nn_q[0];

    for(int i = 1; i < k; i++){
    
        if(dist_nn_q[i] > max){
            max = dist_nn_q[i];
            indice_max = i;
        }
    }
    if(max > dist){
        dist_nn_q[indice_max] = dist;
        id_nn_q[indice_max] = v;
    }
}

void querying(params *input, type *q, int q_index){

    int k = input->k;
    int h = input->h;
    int N = input->N;
    int D = input->D;
    int x = input->x;

    type *DS = input->DS;
    type *Q = input->Q;
    type *index = input->index;
    int *P = input->P;

    int *id_nn_q = input->id_nn + (size_t)q_index * k;
    type *dist_nn_q = input->dist_nn + (size_t)q_index * k;

    // inizializzo i KNN di q a coppie (-1,inf)

    for(int id = 0; id < k; id++){
        id_nn_q[id] = -1;
        dist_nn_q[id] = INFINITY; 
    }

    type *dist_nn_pivot = malloc(h*sizeof(type));   

    if (input->k > input->h) {
        fprintf(stderr, "Errore: k non può essere maggiore di h (k=%d, h=%d)\n", input->k, input->h);
        exit(1);
    }

    if(!dist_nn_pivot){
        fprintf(stderr, "Errore nell'allocazione del vettore per le distanze dai pivot in querying (dist_piv_q)\n");
        exit(1);
    }

    // calcolo delle distanze di q da tutti i pivot

    for(int p = 0; p < h; p++){
        int indice_pivot = P[p];
        type *pivot = DS + (size_t)indice_pivot * D;
        dist_nn_pivot[p] = approx_dist(pivot,q,D,x);
    }

    for(int v = 0; v < N; v++){

        type lower_bound = 0.0;

        //calcolo del lower bound della distanza
        for(int p = 0; p < h; p++){
            type distp = index[v+(size_t)p*N];
            type distq = dist_nn_pivot[p];
            type diff = fabs(distp - distq);
            if(diff > lower_bound){
                lower_bound = diff;
            }
        }

        // selezione del vicino a distanza maggiore
        type dmax = dist_nn_q[0];
        for (int i = 1; i < k; i++) {
            if (dist_nn_q[i] > dmax)
                dmax = dist_nn_q[i];
        }

        if(lower_bound < dmax){
            type *v_vec = DS + (size_t)v*D;
            type dapp = approx_dist(v_vec,q,D,x);
            if (dmax > dapp){
                insert_nn(id_nn_q,dist_nn_q,v,dapp,k); //TODO (controllare l'inserimento in id_nn_q)
            }
        }

    }

    // calcolo della distanza effettiva
    for(int i = 0; i < k; i++){
        int id = id_nn_q[i];
        if(id >= 0){
            type *v_vec = DS + (size_t)id * D;
            type real_dist = distanza(q,v_vec, D); //TODO (controllare il calcolo)
            dist_nn_q[i] = real_dist;
        }
    }

    free(dist_nn_pivot);

}

void stampa_vicini(params* input) {

    // per ogni punto di query q, stampa i suoi k vicini
    for (int i = 0; i < input->nq; i++) {
        type* query     = input->Q       + (size_t)i * input->D;
        int*  neighbours = input->id_nn  + (size_t)i * input->k;
        type* distances  = input->dist_nn + (size_t)i * input->k;

        printf("Per il punto: [");
        for (int j = 0; j < input->D; j++) {
            printf("%.3f", (double)query[j]);
            if (j < input->D - 1)
                printf(", ");
        }
        printf("]\n");

        printf("I suoi %d vicini sono:\n", input->k);

        for (int h = 0; h < input->k; h++) {
            int id_vicino = neighbours[h];

            // puntatore al punto del dataset corrispondente al vicino
            type* punto = input->DS + (size_t)id_vicino * input->D;

            printf("  vicino %d (id %d): [", h, id_vicino);
            for (int j = 0; j < input->D; j++) {
                printf("%.3f", (double)punto[j]);
                if (j < input->D - 1)
                    printf(", ");
            }
            printf("] a distanza: %.3f\n", (double)distances[h]);
        }

        printf("\n");
    }
}


void predict(params* input){
    // Esecuzione delle query

    /*
    input->id_nn[1] = 5;
    prova(input);
    */

    // TODO (allocare le strutture di interesse (id_nn, dist_nn, ...))

    input->id_nn   = malloc((size_t)input->nq * input->k * sizeof(int));
    input->dist_nn = _mm_malloc((size_t)input->nq * input->k * sizeof(type), align);
    

    int nq = input->nq;
    int D = input->D;

    if (!input->silent) {
        printf("Inizio funzione PREDICT: ricerca dei KNN per ciascun punto di query q\n");
    }

    for(int i = 0; i < nq; i++){
        type* q = input->Q + (size_t)i * D;
        querying(input,q,i);
    }

    stampa_vicini(input); // TODO (controllare formattazione)
    
    if (!input->silent) {
        printf("PREDICT completato\n");
    }

    free(input->id_nn);
    _mm_free(input->dist_nn);

}

