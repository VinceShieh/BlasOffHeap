/*CBLAS.c*/ 
#include <jni.h>
#include <assert.h>
#include "mkl_cblas.h"

JNIEXPORT jdouble JNICALL Java_CBLAS_ddotOH(JNIEnv *env, jclass klass, jint N, jdoubleArray x, jint incx, jdoubleArray y, jint incy, jint isCopy) {

    jdouble *xElems, *yElems, res;
    if (isCopy == 1) {
      xElems = (*env)-> GetDoubleArrayElements (env,x,NULL);
      yElems = (*env)-> GetDoubleArrayElements (env,y,NULL);
    } else {
      xElems = (*env)-> GetPrimitiveArrayCritical (env,x,NULL);
      yElems = (*env)-> GetPrimitiveArrayCritical (env,y,NULL);
    }
    assert(xElems && yElems);

    res = cblas_ddot(N, xElems, incx, yElems, incy);
    if (isCopy == 1) {
      (*env)-> ReleaseDoubleArrayElements (env,x,xElems,JNI_ABORT);
      (*env)-> ReleaseDoubleArrayElements (env,y,yElems,JNI_ABORT);
    } else {
      (*env)-> ReleasePrimitiveArrayCritical (env,x,xElems,JNI_ABORT);
      (*env)-> ReleasePrimitiveArrayCritical (env,y,yElems,JNI_ABORT);
    }

   return res;
}

JNIEXPORT void JNICALL Java_CBLAS_dsprOH(JNIEnv *env, jclass klass, jint Order, jint Uplo, jint n, jdouble alpha, jdoubleArray x, jint incx, jlong C, jint isCopy) {
    jdouble *xElems;
    if (isCopy == 1) {
      xElems = (*env)-> GetDoubleArrayElements (env,x,NULL);
    } else {
      xElems = (*env)-> GetPrimitiveArrayCritical (env,x,NULL); 
    }
    assert(xElems);

//    printf("***calling native spr");
    cblas_dspr((CBLAS_ORDER)Order, (CBLAS_UPLO)Uplo, n, alpha, xElems, incx, C);
    if (isCopy == 1) {
      (*env)-> ReleaseDoubleArrayElements (env,x,xElems,JNI_ABORT);
    } else {
      (*env)-> ReleasePrimitiveArrayCritical (env,x,xElems,JNI_ABORT);
    }
}

JNIEXPORT void Java_CBLAS_dgemmFN (JNIEnv *env, jclass klass, jint Order, jint TransA, jint TransB, jint M, jint N, jint K, jdouble alpha, jdoubleArray A, int lda, jdoubleArray B, jint ldb,  jdouble beta,  jlong C, jint ldc, jint isCopy){
    jdouble *aElems, *bElems;
    if (isCopy == 1) {
      aElems = (*env)-> GetDoubleArrayElements (env,A,NULL);
      bElems = (*env)-> GetDoubleArrayElements (env,B,NULL);
    } else {
      aElems = (*env)-> GetPrimitiveArrayCritical (env,A,NULL);
      bElems = (*env)-> GetPrimitiveArrayCritical (env,B,NULL);
    }
    assert(aElems && bElems);
//    printf("*****calling native blas***\n");
    cblas_dgemm ((CBLAS_ORDER)Order,(CBLAS_TRANSPOSE)TransA,(CBLAS_TRANSPOSE)TransB, (int)M,(int)N,(int)K,alpha,aElems,(int)lda,bElems,(int)ldb,beta,(double *)C,(int)ldc);
/*
    int i = 0;
    double * ptr = C;
    while (i<M * N){
	printf("*****return value from native:%f******",ptr[i]);
 	i++;
    }
*/
    if (isCopy == 1) {
      (*env)-> ReleaseDoubleArrayElements (env,B,bElems,JNI_ABORT);
      (*env)-> ReleaseDoubleArrayElements (env,A,aElems,JNI_ABORT);
    } else {
      (*env)-> ReleasePrimitiveArrayCritical (env,B,bElems,JNI_ABORT);
      (*env)-> ReleasePrimitiveArrayCritical (env,A,aElems,JNI_ABORT);
    }
}

JNIEXPORT void Java_CBLAS_dgemm (JNIEnv *env, jclass klass,    jint Order, jint TransA, jint TransB, jint M, jint N, jint K,   jdouble alpha, jdoubleArray A, int lda, jdoubleArray B, jint ldb,  jdouble beta,  jdoubleArray C, jint ldc){
    jdouble *aElems, *bElems, *cElems;
 
    aElems = (*env)-> GetDoubleArrayElements (env,A,NULL);
//    aElems = (*env)->GetDirectBufferAddress(env, A);
    bElems = (*env)-> GetDoubleArrayElements (env,B,NULL);
    cElems = (*env)-> GetDoubleArrayElements (env,C,NULL);
    assert(aElems && bElems && cElems);
 
cblas_dgemm ((CBLAS_ORDER)Order,(CBLAS_TRANSPOSE)TransA,(CBLAS_TRANSPOSE)TransB, (int)M,(int)N,(int)K,alpha,aElems,(int)lda,bElems,(int)ldb,beta,cElems,(int)ldc);
//cblas_dgemm ((CBLAS_ORDER)Order,(CBLAS_TRANSPOSE)TransA,(CBLAS_TRANSPOSE)TransB, (int)M,(int)N,(int)K,alpha,aElems,(int)lda,bElems,(int)ldb,beta,cElems,(int)ldc);
  
   // (*env)-> ReleaseDoubleArrayElements (env,C,cElems,0);
    (*env)-> ReleaseDoubleArrayElements (env,B,bElems,JNI_ABORT);
    (*env)-> ReleaseDoubleArrayElements (env,A,aElems,JNI_ABORT);
    (*env)-> ReleasePrimitiveArrayCritical (env,A,aElems,0);
}

JNIEXPORT void JNICALL Java_CBLAS_dgemv(JNIEnv *env, jclass klass, jint Order, jint TransA, jint M, jint N, jdouble alpha, jdoubleArray A, jint lda, jobject X, jint incx, jdouble beta, jdoubleArray Y, jint incy){
    jdouble *aElems, *xElems, *yElems;
    xElems = (*env)->GetDirectBufferAddress(env, X);
    aElems = (*env)-> GetDoubleArrayElements (env,A,NULL);
    yElems = (*env)-> GetDoubleArrayElements (env,Y,NULL);
    printf("*****calling native blas***\n");
cblas_dgemv ((CBLAS_ORDER)Order,(CBLAS_TRANSPOSE)TransA, (int)M,(int)N,alpha,aElems,(int)lda,xElems,(int)incx,beta,yElems,(int)incy);
    (*env)-> ReleaseDoubleArrayElements (env,Y,yElems,0);
    (*env)-> ReleaseDoubleArrayElements (env,A,aElems,JNI_ABORT);
  //  (*env)-> ReleaseDoubleArrayElements (env,A,aElems,JNI_ABORT);
    (*env)-> ReleasePrimitiveArrayCritical (env,X,xElems,0);
}

JNIEXPORT void JNICALL Java_CBLAS_ttest(JNIEnv *env, jclass klass, jint aa){
    printf("*****calling native blas:%d****\n",aa);
}

JNIEXPORT void JNICALL Java_CBLAS_dmatmul(JNIEnv *env, jclass klass, jint Order, jint TransA, jint TransB, jdouble alpha, jintArray srcIds, jint srcIdsLen, jdoubleArray srcFactors, jint lda, jint rowSrc, jint colSrc, jintArray dstIds, jint dstIdsLen, jdoubleArray dstFactors, jint ldb, jint rowDst, jint colDst, jlong C, jdouble beta, jint isCopy){

    jdouble *srcElems, *dstElems;
    jint *srcIdElems, *dstIdElems;
    if (isCopy == 1) {
      srcElems = (*env)-> GetDoubleArrayElements (env,srcFactors,NULL);
      dstElems = (*env)-> GetDoubleArrayElements (env,dstFactors,NULL);
      srcIdElems = (*env)-> GetIntArrayElements (env,srcIds,NULL);
      dstIdElems = (*env)-> GetIntArrayElements (env,dstIds,NULL);
    } else {
      srcElems = (*env)-> GetPrimitiveArrayCritical (env,srcFactors,NULL);
      dstElems = (*env)-> GetPrimitiveArrayCritical (env,dstFactors,NULL);
      srcIdElems = (*env)-> GetPrimitiveArrayCritical (env,srcIds,NULL);
      dstIdElems = (*env)-> GetPrimitiveArrayCritical (env,dstIds,NULL);
      
    }
	
//    double C[rowSrc * colDst]; // change to 2-D array?? 
//    double C[rowSrc][colDst]; // change to 2-D array?? 
//    memset(C,0,sizeof(C));
    cblas_dgemm ((CBLAS_ORDER)Order,(CBLAS_TRANSPOSE)TransA,(CBLAS_TRANSPOSE)TransB, (int)rowSrc,(int)colDst,(int)colSrc,alpha,srcElems,(int)lda,dstElems,(int)ldb,beta,C,(int)rowSrc);
    
  /*  
    struct outRank{
	int srcID;
	int dstID;
	double r;
    };
*/
    int i,j,k=rowSrc*colDst;
    int *ptr;
    ptr =(long) C + k;
    for(i=0; i < rowSrc; i++){
	for(j = 0; j < colDst; j++){
	    *ptr = srcIdElems[i];
	    ptr++;
	    *ptr = dstIdElems[j];
	    ptr++; 
	}
    }
    if (isCopy == 1) {
      (*env)-> ReleaseDoubleArrayElements (env,srcFactors,srcElems,JNI_ABORT);
      (*env)-> ReleaseDoubleArrayElements (env,dstFactors,dstElems,JNI_ABORT);
      (*env)-> ReleaseIntArrayElements (env,srcIds,srcIdElems,JNI_ABORT);
      (*env)-> ReleaseIntArrayElements (env,dstIds,dstIdElems,JNI_ABORT);
    } else {
      (*env)-> ReleasePrimitiveArrayCritical (env,dstIds,dstIdElems,0);
      (*env)-> ReleasePrimitiveArrayCritical (env,srcIds,srcElems,0);
      (*env)-> ReleasePrimitiveArrayCritical (env,dstFactors,dstElems,0);
      (*env)-> ReleasePrimitiveArrayCritical (env,srcFactors,srcElems,0);
    }
}
