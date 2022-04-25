#include <CL/sycl.hpp>                                                                                                                                    
using namespace sycl;
#define WRAP_SIZE 32
int main(){
        queue q;
        int num_blocks=128;
        int num_threads=256;
        q.submit([&](sycl::handler& cgh){
                sycl::stream out{ 4096, 128, cgh };
                 cgh.parallel_for(
                        sycl::nd_range<1>(num_blocks * num_threads, num_threads),
                        [=](sycl::nd_item<1> item_ct1) [[intel::reqd_sub_group_size(WRAP_SIZE)]] {
                                int blkId = item_ct1.get_group(0);
                                int tid = item_ct1.get_local_id(0);
                                auto sg = item_ct1.get_sub_group();
                                int warpSize = sg.get_local_range()[0];
                                if(tid<warpSize){
                                        sg.barrier();
                                        out<<"sub group sync\n";
                                }
                        }
                );
        });
}