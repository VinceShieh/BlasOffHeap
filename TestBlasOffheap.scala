import com.github.fommil.netlib.F2jBLAS
import com.github.fommil.netlib.BLAS.{getInstance => nativeBlas}
import scala.util.Random
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.unsafe.Platform

object TestBlasOffheap{
  @transient var f2j = new F2jBLAS
  def main(args: Array[String]): Unit = {
	val conf = new SparkConf().setAppName("Test Blas offheap")
	val sc = new SparkContext(conf)
//BLAS-level 3
/*
	testGemm(10000,5000,5000,2000,10,1)
	testGemm(10000,5000,5000,2000,100,1)
	testGemm(10000,5000,5000,2000,1000,1)
	testGemm(10000,5000,5000,2000,10000,1)
	testGemm(10000,5000,5000,5000,10000,1)
	testGemm(20000,5000,5000,10000,10000,1)
//	testGemm(100000,5000,5000,100000,10000,0)


//BLAS-level 2
	testDspr(1000,1000,0)
	testDspr(10000,10000,0)
	testDspr(20000,20000,0)
*/
//BLAS-level 1
	testDot(100000,0)
	testDot(1000000,0)
	testDot(1400000,0)
	testDot(2000000,0)
	testDot(3000000,0)
	testDot(10000000,0)
	testDot(100000000,0)
	testDot(1000000000,0)
//	testDot(2000000000,0)

	sc.stop()
  }

// ddot - x*y
  def testDot(numElem: Int, isCopy: Int): Unit = {
//	CBLAS.loadtt()
	
	val x = generateRandomArray(numElem)
	val y = generateRandomArray(numElem)
	var timeStart = System.nanoTime()
	
//	val a = CBLAS.ddotOH(numElem, x, 1, y, 1, isCopy)
        val a = nativeBlas.ddot(numElem, x, 1, y, 1)

	val timeMKL = (System.nanoTime() - timeStart) / 1e9

	timeStart = System.nanoTime()
	
	val b = f2j.ddot(numElem, x, 1, y, 1)

	val timeF2j = (System.nanoTime() - timeStart) / 1e9

	println(s" ddot total time(s) compute $numElem number of data for MKL isCopy:$isCopy is: $timeMKL, F2J is: $timeF2j, MKL return:$a, F2J return:$b, no offheap data retrieve needed!")

  }

// spr - Adds alpha*x*xt to matrix U in-place
  def testDspr(numElem: Int, resNum: Int, isCopy: Int): Unit = {
	CBLAS.loadtt()
	val Order: Int = CBLAS.ORDER.ColMajor
	val uplo: Int = CBLAS.UPLO.Upper
	val alpha = 0.1

	val x = generateRandomArray(numElem)
	var timeStart = System.nanoTime()

	val mem = Platform.allocateMemory(numElem*(numElem+1)/2 * 8)
	CBLAS.dsprOH(Order, uplo, numElem, alpha, x, 1, mem, isCopy)

        var t1 = System.nanoTime()
	var timeCompMKL = (t1 - timeStart) / 1e9

	val resmkl = getArrayNative(mem, resNum)

	var timeRetrieveMKL = (System.nanoTime() - t1) / 1e9
	Platform.freeMemory(mem)

	val timeMKL = (System.nanoTime() - timeStart) / 1e9

	timeStart = System.nanoTime()

	val U = generateRandomArray(numElem*(numElem+1)/2)
	f2j.dspr("U",numElem, alpha, x, 1, U)

        t1 = System.nanoTime()
	var timeCompF2j = (t1 - timeStart) / 1e9

	val resF2j = {
	  var i = 0
	  var resArr = new Array[Double](0)
	  while (i < resNum) {
	    resArr = resArr :+ U(i)
//	    println("***get f2j data:" + U(i))
	    i = i + 1
	  }
	  resArr
	}

	var timeRetrieveF2j = (System.nanoTime() - t1) / 1e9
	val timeF2j = (System.nanoTime() - timeStart) / 1e9

	println(s" spr total time(s) retrieve $resNum number of data for MKL isCopy: $isCopy is: $timeMKL, F2J is: $timeF2j,Details:mkl compute time:$timeCompMKL, data retrieve time:$timeRetrieveMKL, F2J compute time:$timeCompF2j, data retrive time:$timeRetrieveF2j")

  }

// gemm - C = alpha*A*B + C
  def testGemm(Am: Int, An: Int, Bm: Int, Bn: Int, resNum: Int, isCopy: Int) = {
	assert(resNum <= Am * Bn)

	CBLAS.loadtt()

	val Order: Int = CBLAS.ORDER.ColMajor
	val TransA: Int = CBLAS.TRANSPOSE.NoTrans
	val TransB: Int = CBLAS.TRANSPOSE.NoTrans
	val alpha: Double = 1.0 
	val beta: Double = 0.0

	val A = generateRandomMatrix(Am, An)
	val B = generateRandomMatrix(Bm, Bn)
	val lda: Int = A.numRows
	val ldb: Int = A.numCols
	val ldc: Int = A.numRows
	
//Offheap MKL
	var timeStart = System.nanoTime()

        val mem = Platform.allocateMemory(Am * Bn * 8)
	CBLAS.dgemmFN (Order, TransA, TransB, A.numRows, B.numCols, A.numCols, alpha, A.values, lda, B.values, ldb, beta, mem, ldc, isCopy)
        
	var t1 = System.nanoTime()
	var timeCompMKL = (t1 - timeStart) / 1e9

	val resmkl = getArrayNative(mem, resNum)
	
	var timeRetrieveMKL = (System.nanoTime() - t1) / 1e9
	Platform.freeMemory(mem)

	val timeMKL = (System.nanoTime() - timeStart) / 1e9
//F2J Blas
	timeStart = System.nanoTime()
	val C = DenseMatrix.zeros(Am, Bn)
	f2j.dgemm("N", "N", A.numRows, B.numCols, A.numCols, alpha, A.values, lda, B.values, ldb, beta, C.values, ldc)

        t1 = System.nanoTime()
	var timeCompF2j = (t1 - timeStart) / 1e9

	val resF2j = {
	  var i = 0
	  var resArr = new Array[Double](0)
	  while (i < resNum) {
	    resArr = resArr :+ C.values(i)
//	    println("***get f2j data:" + C.values(i))
	    i = i + 1
	  }
	  resArr
	}
	var timeRetrieveF2j = (System.nanoTime() - t1) / 1e9
	val timeF2j = (System.nanoTime() - timeStart) / 1e9

	println(s" gemm total time(s) on matrix ($Am,$An)x($Bm,$Bn) retrieve $resNum for MKL isCopy:$isCopy is: $timeMKL, F2J is: $timeF2j,Details:mkl compute time:$timeCompMKL, data retrieve time:$timeRetrieveMKL, F2J compute time:$timeCompF2j, data retrive time:$timeRetrieveF2j")

/*
//netlib MKL
        var timeStart = System.nanoTime()
	val CNet = DenseMatrix.zeros(Am, Bn)
	nativeBlas.dgemm("N", "N", A.numRows, B.numCols, A.numCols, alpha, A.values, lda, B.values, ldb, beta, CNet.values, ldc)
	var t1 = System.nanoTime()
	var timeNetMKL = (t1 - timeStart) / 1e9
	val resNetlib = {
	  var j = 0
	  var netArr = new Array[Double](0)
	  while (j < resNum) {
	    netArr = netArr :+ CNet.values(j)
//	    println("***get netlib MKL data:" + CNet.values(i))
	    j = j + 1
	  }
	  netArr
	}
	var timeRetrieveNetMKL = (System.nanoTime() - t1) / 1e9
	val timeNet = (System.nanoTime() - timeStart) / 1e9
	println(s" gemm time(s) on matrix ($Am,$An)x($Bm,$Bn) retrieve $resNum number of data for netlib MKL is $timeNet, Details: compute:$timeNetMKL, data retrieve:$timeRetrieveNetMKL")
*/
  }

  def getArrayNative(addr: Long, num: Int) : Array[Double] = {
	var ptr = addr
	var i = 0
	var resArr = new Array[Double](0)
	while (i < num) {
	  val data = Platform.getDouble(null, ptr)
//	  println("***get offheap data:" + data)
	  resArr = resArr :+ data
	  ptr += 8
	  i += 1
	}
	resArr
  }

  def generateRandomArray(len: Int): Array[Double] = {
	(for (i <- 1 to len) yield Random.nextDouble).toArray
  }
  def generateRandomMatrix(m: Int, n: Int): DenseMatrix = {
	val arrayValue = generateRandomArray(m*n)
	new DenseMatrix(m, n, arrayValue)
  }

}
