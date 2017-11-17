//package nativeBLASV;
import java.nio.ByteBuffer;
public final class CBLAS {
/* private CBLAS() {}
   static {
     System.loadLibrary("mkl_java_stubs"); 
   }*/
public static void loadtt(){
   {
    try{
      String wrapper = System.getProperty("com.intel.mkl", "libmkl_java_stubsso");
      System.load(wrapper);
/*
String javaLibPath = System.getProperty("java.library.path");
System.out.println(javaLibPath);
	System.loadLibrary("mkl_als_stubs");    
*/

  //    System.out.println("Native code library loaded.\n");
    } catch (UnsatisfiedLinkError e) {
      System.out.println("Native code library failed to load.\n" + e);
      System.exit(1);
    }
  }
}
   public final static class arrayTuple {
     public arrayTuple(int a, int b, double c) {
	srcIds = a;
	dstIds = b;
	r = c;
     }
     public int srcIds;
     public int dstIds;
     public double r;
   }
   public final static class ORDER {
     private ORDER() {}
     /** row-major arrays */
     public final static int RowMajor=101;
     /** column-major arrays */
     public final static int ColMajor=102;
   }
   public final static class TRANSPOSE {
     private TRANSPOSE() {}
     /** trans='N' */
     public final static int NoTrans =111;
     /** trans='T' */
     public final static int Trans=112;
     /** trans='C' */
     public final static int ConjTrans=113;
  }
   public final static class UPLO {
     private UPLO() {}
     public final static int Upper=121;
     public final static int Lower=122;
   }
  public static native void dsprOH(int Order, int Uplo, int N, double alpha, double[] x, int incx, long C, int isCopy);
  public static native double ddotOH(int N, double[] x, int incx, double[] y, int incy, int isCopy);
  public static native void dgemm(int Order, int TransA, int TransB, int M, int N, int K, double alpha, double[] A, int lda, double[] B, int ldb, double beta, double[] C, int ldc); /*inform java virtual machine that function is defined externally*/
  public static native void dmatmul(int Order, int TransA, int TransB, double alpha, int[] srcIds, int srcIdsLen, double[] srcFactors, int lda, int rowSrc, int colSrc, int[] dstIds, int dstIdsLen, double[] dstFactors, int ldb, int rowDst, int colDst, long C, double beta, int isCopy);
  public static native void dgemmFN(int Order, int TransA, int TransB, int M, int N, int K, double alpha, double[] A, int lda, double[] B, int ldb, double beta, long C, int ldc, int isCopy); /*inform java virtual machine that function is defined externally*/


    public static native void dgemv(int Order, int TransA, int M, int N, double alpha, double[] A, int lda, ByteBuffer X, int incx, double beta, double[] Y, int incy); /*inform java virtual machine that function is defined externally*/
//  public static native void dgemv(int Order, int TransA, int M, int N, double alpha, double[] A, int lda, double[] X, int incx, double beta, double[] Y, int incy); /*inform java virtual machine that function is defined externally*/
  public static native void ttest(int aa);
  public static void tt1(arrayTuple[] bb){
	System.out.println("****tt1:****" + bb[0].srcIds);
	ttest(5);
	}
}
