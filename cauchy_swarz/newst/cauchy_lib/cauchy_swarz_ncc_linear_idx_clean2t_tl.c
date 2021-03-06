#include "cauchy_swarz_ncc_match.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define SEC_RES 1000

int cauchy_swarz_ncc_linear_idx_clean2t_tl(const REAL_TYPE* I,const REAL_TYPE* I2, const REAL_TYPE* img,int img_r,int img_c,const REAL_TYPE* pat,int n_p,int m_p,REAL_TYPE* norms_of_wins_minus_mu,REAL_TYPE* wins_mu,const int* box_arr,const REAL_TYPE* w_arr,int box_num,const REAL_TYPE* thresholds,const REAL_TYPE* residual_pat_norms, REAL_TYPE frac_for_direct,int* lidx, REAL_TYPE* vals, REAL_TYPE* U,int* iters_num_ptr,int tl){
    if(tl>0){
        printf("(%f) ",U[tl]);
    }
    int c_r=img_r-n_p+1;
    int c_c=img_c-m_p+1;   
    int indexes_num = linear_idx_norms_mu_U_before_loop_find(I,I2,img_r+1,U,box_arr[0]-1,box_arr[1]-1,box_arr[2]-1,box_arr[3]-1,w_arr[0],norms_of_wins_minus_mu,img_r,img_c,n_p,m_p, wins_mu,lidx,thresholds[0]*residual_pat_norms[0]-residual_pat_norms[1]);
    if(tl>0){        REAL_TYPE u =(U[tl]+residual_pat_norms[1])/residual_pat_norms[0];
        printf("(%f %f) ",u,thresholds[0]);
        printf("(w %f  n %f) ",wins_mu[tl],norms_of_wins_minus_mu[tl]);
        int from_r=box_arr[0]-1;
        int to_r=box_arr[1]-1;
        int from_c=box_arr[2]-1;
        int to_c=box_arr[3]-1;
        int ssz=    (to_r-from_r+1)*(to_c-from_c+1);
        printf("( tmp*w=%f) ",U[tl]*(norms_of_wins_minus_mu[tl])+wins_mu[tl]*ssz*w_arr[0]);
        printf("( w=%f) ",w_arr[0]);
    }
    
    box_arr=box_arr+4;
    int total_indexes_num=c_r*c_c;
    REAL_TYPE cur_frac = ((REAL_TYPE) indexes_num)/((REAL_TYPE) total_indexes_num);
    int t=1;
    while(t<box_num && cur_frac>frac_for_direct){
        int f_r=(*box_arr)-1;
        int t_r=*(box_arr+1)-1;
        int f_c=*(box_arr+2)-1;
        int t_c=*(box_arr+3)-1;
        box_arr=box_arr+4;
        if(cur_frac<0.5){
            indexes_num = lidx_ncc_inside_loop(I,img_r+1,U,norms_of_wins_minus_mu,c_r,c_c,f_r,t_r,f_c,t_c,w_arr[t],wins_mu,lidx,indexes_num,thresholds[t]*residual_pat_norms[0]-residual_pat_norms[t+1]);       
        }
        else{
            indexes_num = lidx_ncc_inside_loop(I,img_r+1,U,norms_of_wins_minus_mu,c_r,c_c,f_r,t_r,f_c,t_c,w_arr[t],wins_mu,lidx,indexes_num,thresholds[t]*residual_pat_norms[0]-residual_pat_norms[t+1]);
        }
        if(tl>0){
            REAL_TYPE u =(U[tl]+residual_pat_norms[t+1])/residual_pat_norms[0];
            printf("(%f %f) ",u,thresholds[t]);
        }
        cur_frac = ((REAL_TYPE) indexes_num)/((REAL_TYPE) total_indexes_num);
        t++;
    }
    
    if(tl>0){
        printf("\n");
    }
    *iters_num_ptr=t-1;
    lidx_direct_ncc(img,img_r,img_c,pat,n_p,m_p, lidx,vals,indexes_num,residual_pat_norms[0],norms_of_wins_minus_mu);
    return indexes_num;
}



