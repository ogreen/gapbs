// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>

#include <omp.h>
#include <pthread.h>

#include <algorithm>
#include <vector>



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
  pvector<ScoreT> scores(g.num_nodes(), init_score);
  pvector<ScoreT> outgoing_contrib(g.num_nodes());
  pvector<NodeID> lrb_queue(g.num_nodes());
  pvector<NodeID> lrb_sizes(g.num_nodes());
  pvector<NodeID> lrb_pos(g.num_nodes());


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

	    int baa = (g.num_nodes() / nthreads) * thread_id;
	    int ele = g.num_nodes() / nthreads;
	    for (NodeID u=0; u < ele; u++){
	    	int newpos=u*nthreads+thread_id;
	    	if(newpos<g.num_nodes())
		    	lrb_pos[u+baa]=newpos;
	    }
  }



	for (int iter=0; iter < max_iters; iter++) {
		double error = 0;
		#pragma omp parallel for
		for (NodeID n=0; n < g.num_nodes(); n++)
		  outgoing_contrib[n] = scores[n] / g.out_degree(n);
		  #pragma omp parallel for
		  for (NodeID n=0; n < g.num_nodes(); n++){
 		  	NodeID pos = lrb_pos[n];
		    NodeID u = lrb_queue[pos]; 
		    ScoreT incoming_total = 0;

			for (int d=0; d< g.in_degree(u); d++){
				NodeID v = g.in_index_[u][d];
		     	incoming_total += outgoing_contrib[v];
			}
		    // for (NodeID v : g.in_neigh(u))
		    //   incoming_total += outgoing_contrib[v];


		// ScoreT old_score = scores[u];
		    scores[u] = base_score + kDamp * incoming_total;
		    // scores[n] = base_score + kDamp * incoming_total;
		  }
	}


 //  int32_t lrb_bins_global[32];
 //  int32_t lrb_prefix_global[33];
 //  int32_t currSize;

 //  double errorTotal=0;
 //  int32_t threads;
 //  #pragma omp parallel
 //  {
 //    int32_t lrb_bins_local[32];
 //    int32_t lrb_pos_local[32];

 //    omp_set_num_threads (36);
 //    #pragma omp barrier


 //    int32_t nthreads = omp_get_num_threads ();
 //    int32_t thread_id = omp_get_thread_num ();

	// NodeID start = (g.num_nodes() / nthreads) * thread_id;
	// NodeID end   = (g.num_nodes() / nthreads) * (thread_id + 1);
	// if (thread_id == 0)  start = 0;
	// if (thread_id == nthreads - 1) end = g.num_nodes();

 //    #pragma omp single
 //    {
 //      for(int l=0; l<32; l++)
 //        lrb_bins_global[l]=0;        
	// 	threads=nthreads;
 //    }
 //    for(int l=0; l<32; l++)
 //      lrb_bins_local[l]=0;        

 //    // #pragma omp for
 //    // for (NodeID u = 0; u < g.num_nodes(); u++) {
 //    for (NodeID u=start; u < end; u++){
 //        lrb_sizes[u] = 32 - __builtin_clz((uint32_t)g.out_degree(u));
 //        lrb_bins_local[lrb_sizes[u]]++;
 //    }

 //    for(int l=0; l<32; l++){
 //      __sync_fetch_and_add(lrb_bins_global+l, lrb_bins_local[l]);
 //    }

 //    #pragma omp barrier

 //    #pragma omp single
 //    {
 //      int32_t lrb_prefix_temp[33];
 //      lrb_prefix_temp[32]=0;

 //      for(int l=31; l>=0; l--){
 //        lrb_prefix_temp[l]=lrb_prefix_temp[l+1]+lrb_bins_global[l];
 //      }
 //      for(int l=0; l<32; l++){
 //        lrb_prefix_global[l]=lrb_prefix_temp[l+1];
 //      }
 //    }

 //    #pragma omp barrier

 //    for(int l=0; l<32; l++){
 //      lrb_pos_local[l] = __sync_fetch_and_add(lrb_prefix_global+l, lrb_bins_local[l]);
 //    }

 //    // #pragma omp for
 //    // for (NodeID u = 0; u < g.num_nodes(); u++) {
 //    for (NodeID u=start; u < end; u++){
 //        lrb_queue[lrb_pos_local[lrb_sizes[u]]]=u;
 //        lrb_pos_local[lrb_sizes[u]]++;
 //    }

 //    int baa = (g.num_nodes() / nthreads) * thread_id;
 //    int ele = g.num_nodes() / nthreads;
 //    for (NodeID u=0; u < ele; u++){
 //    	int newpos=u*nthreads+thread_id;
 //    	if(newpos<g.num_nodes())
	//     	lrb_pos[u+baa]=newpos;
 //    }
 //  }

 //  // for (int i=0; i<40; i++)
 //  // 	printf("%d, ", lrb_pos[i]);


   

 //    for (int iter=0; iter < max_iters; iter++) {
 
 //      #pragma omp parallel
 //      for (NodeID n=0; n < g.num_nodes(); n++)
 //        outgoing_contrib[n] = scores[n] / g.out_degree(n);

 //      #pragma omp parallel
 //      for (NodeID n=0; n < g.num_nodes(); n++){
 // 	    // NodeID pos=n/omp_get_num_threads;
 // 	    // NodeID pos=(n*threads)/g.num_nodes();
 //      	NodeID pos = lrb_pos[n];
 //        NodeID u = lrb_queue[pos]; 
 //        ScoreT incoming_total = 0;
 //        for (NodeID v : g.in_neigh(u))
 //          incoming_total += outgoing_contrib[v];
 //  		ScoreT old_score = scores[u];
 //        scores[u] = base_score + kDamp * incoming_total;
 //      }

 //    }


  return scores;
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
