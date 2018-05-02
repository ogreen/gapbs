// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>

#include <omp.h>
#include <pthread.h>

#include <algorithm>
#include <vector>

#include <immintrin.h>



#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"


/*
GAP Benchmark Suite
Kernel: PageRank (PR)
Author: Scott Beamer

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. This is done
to ease comparisons to other implementations (often use same algorithm), but
it is not necesarily the fastest way to implement it. It does perform the
updates in the pull direction to remove the need for atomics.
*/


using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;

pvector<ScoreT> PageRankPull(const Graph &g, int max_iters,
                             double epsilon = 0) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> scoresVec(g.num_nodes(), init_score);
  // pvector<NodeID> lrb_queue(g.num_nodes());
  NodeID* lrb_queue = new NodeID[g.num_nodes()];
  ScoreT* scores = new ScoreT[g.num_nodes()];
  ScoreT* outgoing_contrib = new ScoreT[g.num_nodes()];

  pvector<NodeID> lrb_sizes(g.num_nodes());
  pvector<NodeID> lrb_pos(g.num_nodes());

  pvector<SGOffset> offSetVector = g.VertexOffsets(true);
  NodeID* offSet = new NodeID[g.num_nodes()+1];


  int32_t lrb_bins_global[32];
  int32_t lrb_prefix_global[33];

  int32_t lrb_bins_local_array[256][32];
  int32_t lrb_pos_local_array[256][32];

  // double errorTotal=0;
  int32_t nthreads;

	#pragma omp parallel
    {
  		nthreads = omp_get_num_threads();
  		#pragma omp barrier
	}

    #pragma omp parallel for
    for (int n=0; n < g.num_nodes(); n++)
      scores[n] = init_score;


    #pragma omp parallel for
    for (int n=0; n < g.num_nodes()+1; n++)
      offSet[n] = offSetVector[n]; 


	#pragma omp parallel for
	for (int thread_id=0; thread_id < nthreads; thread_id++){
	    int32_t* lrb_bins_local = lrb_bins_local_array[thread_id];
	    // int32_t* lrb_pos_local  = lrb_pos_local_array[thread_id];

		if (thread_id==0){
			for(int l=0; l<32; l++)
				lrb_bins_global[l]=0;
		// threads=nthreads;
		}
	    for(int l=0; l<32; l++)
	      lrb_bins_local[l]=0;        

		NodeID start = (g.num_nodes() / nthreads) * thread_id;
		NodeID end   = (g.num_nodes() / nthreads) * (thread_id + 1);
		if (thread_id == 0)  start = 0;
		if (thread_id == nthreads - 1) end = g.num_nodes();

	    for (NodeID u=start; u < end; u++){
	        lrb_sizes[u] = 32 - __builtin_clz((uint32_t)g.in_degree(u));
	        lrb_bins_local[lrb_sizes[u]]++;
	    }

	    for(int l=0; l<32; l++){
	      __sync_fetch_and_add(lrb_bins_global+l, lrb_bins_local[l]);
	    }
	}  

    int32_t lrb_prefix_temp[33];
    lrb_prefix_temp[32]=0;
    for(int l=31; l>=0; l--){
    	lrb_prefix_temp[l]=lrb_prefix_temp[l+1]+lrb_bins_global[l];
    }
    for(int l=0; l<32; l++){
      	lrb_prefix_global[l]=lrb_prefix_temp[l+1];
    }    

	#pragma omp parallel for
	for (int thread_id=0; thread_id < nthreads; thread_id++){
	    int32_t* lrb_bins_local = lrb_bins_local_array[thread_id];
	    int32_t* lrb_pos_local  = lrb_pos_local_array[thread_id];

	    for(int l=0; l<32; l++){
	     	lrb_pos_local[l] = __sync_fetch_and_add(lrb_prefix_global+l, lrb_bins_local[l]);
	    }

		NodeID start = (g.num_nodes() / nthreads) * thread_id;
		NodeID end   = (g.num_nodes() / nthreads) * (thread_id + 1);
		if (thread_id == 0)  start = 0;
		if (thread_id == nthreads - 1) end = g.num_nodes();

	    for (NodeID u=start; u < end; u++){
	        lrb_queue[lrb_pos_local[lrb_sizes[u]]]=u;
	        lrb_pos_local[lrb_sizes[u]]++;
	    }

	    // int baa = (g.num_nodes() / nthreads) * thread_id;
	    // int ele = g.num_nodes() / nthreads;
	    // for (NodeID u=0; u < ele; u++){
	    // 	int newpos=u*nthreads+thread_id;
	    // 	if(newpos<g.num_nodes())
		   //  	lrb_pos[u+baa]=newpos;
	    // }
    }

	static const __m512i mizero32 = _mm512_set_epi32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
	static const  __m512i mione32 = _mm512_set_epi32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1); 
	static const  __m512 mfzero32 = _mm512_set_ps(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0); 
	for (int iter=0; iter < max_iters; iter++) {
		int step = nthreads*16;

		#pragma omp parallel for
		for (NodeID n=0; n < g.num_nodes(); n++)
		  outgoing_contrib[n] = scores[n] / g.out_degree(n);

		#pragma omp parallel for
  		for (int thread_id=0; thread_id < nthreads; thread_id++){


			__m512 baseVec = _mm512_set_ps(base_score,base_score,base_score,base_score,base_score,base_score,base_score,base_score,
										   base_score,base_score,base_score,base_score,base_score,base_score,base_score,base_score);

			__m512 kDampVec = _mm512_set_ps(kDamp,kDamp,kDamp,kDamp,kDamp,kDamp,kDamp,kDamp,
											kDamp,kDamp,kDamp,kDamp,kDamp,kDamp,kDamp,kDamp);
  		    for (int32_t pos = thread_id*16; pos < g.num_nodes(); pos+=step) {

			    if((pos+16)>=g.num_nodes()){
			    	for(int32_t pos2=pos; pos2<g.num_nodes(); pos2++){
					    NodeID u = lrb_queue[pos2]; 

					    ScoreT incoming_total = 0;
						for (int d=0; d< g.in_degree(u); d++){
							NodeID v = g.in_index_[u][d];
					     	incoming_total += outgoing_contrib[v];
						}
					    scores[u] = base_score + kDamp * incoming_total;
			    	}
			    	break;
			    }

				int32_t exceededAStop,maskA;

			    __m512i uVec 		   		= _mm512_load_epi32(lrb_queue+pos);
				__m512i uVecP1         		= _mm512_add_epi32(uVec,mione32);		    
				__m512 incoming_totalVec	= mfzero32;

			    __m512i indexA       = _mm512_i32gather_epi32(uVec, offSet, 4);
			    __m512i indexAStop	= _mm512_i32gather_epi32(uVecP1, offSet, 4);
			 	exceededAStop 		= _mm512_cmpgt_epi32_mask(indexAStop, indexA);

	    		// for(int32_t p=0;p<16; p++){
	    		// 	uint32_t *val1 = (uint32_t*) &indexAStop;
	    		// 	uint32_t *val2 = (uint32_t*) &indexA;
	    		// 	if((val1[p]-val2[p])<0)
		    	// 		printf("OUCH");
	    		// }
	    		// for(int32_t p=pos;p<(pos+16); p++){
	    		// 	if((offSet[lrb_queue[p]+1]-offSet[lrb_queue[p]])<0)
	    		// 		printf("%d,",lrb_queue[p]);
		    	// 	// printf("(%d, %d) ",offSet[lrb_queue[p]+1]-offSet[lrb_queue[p]],g.in_degree(lrb_queue[p]));
	    		// }
	    		// printf("\n");

				__m512i miAelems;__m512  mContri;
			    while (exceededAStop != 0) {
			        miAelems			= _mm512_mask_i32gather_epi32(miAelems, exceededAStop,indexA,(const int32_t *)g.in_neighbors_, 4);
			     	mContri  			= _mm512_mask_i32gather_ps(mContri, exceededAStop ,miAelems, outgoing_contrib,4);

			     	incoming_totalVec 	= _mm512_mask_add_ps(incoming_totalVec,exceededAStop,incoming_totalVec, mContri);

			     	indexA    			= _mm512_mask_add_epi32(indexA, exceededAStop, indexA,mione32);
			        exceededAStop 		= _mm512_cmpgt_epi32_mask(indexAStop, indexA);
			 //        // break;
			    
			    }

				incoming_totalVec = _mm512_fmadd_ps(kDampVec,incoming_totalVec,baseVec);
				// scores[u] = base_score + kDamp * incoming_total;

			    _mm512_i32scatter_ps(scores,uVec,incoming_totalVec,4);

		  }
		}
	}

	// printf("EXITED\n"); fflush(stdout);

    #pragma omp parallel for
    for (int n=0; n < g.num_nodes(); n++)
      scoresVec[n] = scores[n];


	delete[] lrb_queue;
	delete[] scores;
	delete[] outgoing_contrib;
	delete[] offSet;

 

  return scoresVec;
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 10;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incomming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
    incomming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}


int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  auto PRBound = [&cli] (const Graph &g) {
    return PageRankPull(g, cli.max_iters(), cli.tolerance());
  };
  auto VerifierBound = [&cli] (const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}
