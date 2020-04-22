package com.github.unsupervise.spark.tss

import breeze.linalg.DenseVector
import breeze.math.Complex
import breeze.signal.{fourierTr, iFourierTr}
import org.apache.spark.ml.linalg.Vectors
import smile.wavelet

package object functions {

  /**
    * Fast Fourier Transform lambda, taking a Seq[Double] storing the input time series with constant time granularity.
    * This function uses Breeze fourierTr for actual FFT computation
    */
  val fft: Seq[Double] => Seq[Double] = ts => {
    val dft = fourierTr(new DenseVector(ts.toArray))
    //Transform Complex Vector into Double Array of twice the size
    //NOTE: Solution with while loop WAY FASTER than map > reduce
    var arrayRes = Array.fill(dft.length * 2)(0D)
    var i = 0
    while (i < dft.length) {
      arrayRes(i * 2) = dft(i).real
      arrayRes(i * 2 + 1) = dft(i).imag
      i += 1
    }
    arrayRes.toSeq
  }

  /**
    * Discrete Fourier Transform Periodogram lambda, taking a Seq[Double] storing the input DFT of a time series.
    * The Periodogram consists in complex modulus of input DFT complex values, flattened as a sequence of Double.
    */
  val dftPeriodogram = (dft: Seq[Double]) => {
    val len = dft.size
    val len2 = len / 2
    (0 until len by 2).map(i => {
      (Math.pow(dft(i), 2) + Math.pow(dft(i + 1), 2)) / len2
    })
  }

  /**
    * Inverse Fast Fourier Transform lambda, taking a Seq[Double] storing the input time series with constant time granularity
    * or DFT. This function uses Breeze iFourierTr for actual FFT computation
    */
  val inverseFFT = (ts: Seq[Double]) => {
    val length = ts.size
    val completedFT = (0 until length by 2).map{ j =>
      Complex(ts(j), ts(j + 1))
    }.toArray
    val ifft = iFourierTr(new DenseVector(completedFT))
    ifft.map(fc => fc.real).toArray
  }

  /**
    * Discrete Wavelet Transform lambda, taking a Seq[Double] storing the input time series with constant time granularity.
    * This functions uses scala SMILE implementation for actual DWT computation
    * @param filter the wavelet family name, as supported by SMILE library
    */
  def dwt(filter: String) = (ts: Seq[Double]) => {
    val tsa = ts.toArray
    val wvlet = wavelet.wavelet(filter)
    wvlet.transform(tsa)
    tsa
  }

  /**
    * Inverse Discrete Wavelet Transform lambda, taking a Seq[Double] storing dwt to convert back to time series.
    * This functions uses scala SMILE implementation for actual Inverse DWT computation
    * @param filter the wavelet family name, as supported by SMILE library
    */
  def idwt(filter: String) = (ts: Seq[Double]) => {
    val tsa = ts.toArray
    val wvlet = wavelet.wavelet(filter)
    wvlet.inverse(tsa)
    tsa
  }

  /**
    * Log10 computation over a sequence lambda
    * @param safetyOffset the offset to add to avoid log10(0) errors
    * @return a lambda taking a Seq[Double] and returning another Seq[Double] with same number of elements, each element
    *         being the log10(x + <safetyOffset>) of its counterpart in input Seq.
    */
  def log10(safetyOffset: Double) = (ts: Seq[Double]) => {
    ts.map(x => Math.log10(x + safetyOffset))
  }

  /**
    * Lambda to center a Seq[Double] so that output Seq[Double] mean is 0.
    */
  val center = (seq: Seq[Double]) => {
    val mean = seq.sum / seq.size
    seq.map(_ - mean)
  }

  /**
    * Lambda to weight a Seq[Double]
    */
  val weighted = (seq: Seq[Double], weights: Seq[Double]) => {
    seq.zip(weights).map(x => x._1 * x._2)
  }

  /**
    * Lambda to scale a Seq[Double] so that output Seq[Double] standard deviation is 1.
    * @param zeroPrecision a threshold below which a value is considered 0
    * @return a scale lambda for given precision
    */
  def scale(zeroPrecision: Double) = (seq: Seq[Double]) => {
    val mean = seq.sum / seq.size
    val std = Math.sqrt(seq.map(x => Math.pow(x - mean, 2D)).sum / seq.size)
    if((std - 0D).abs < zeroPrecision) seq.map(x => 0D)
    else seq.map(_ / std)
  }

  /**
    * Lambda to normalize a Seq[Double] so that the output Seq[Double] sum is 1.
    * @return a sum to 1 normalization lambda of Seq[Double] inputs
    */
  def normalizeSum = (seq: Seq[Double]) => {
    val sum = seq.sum
    seq.map(_ / sum)
  }

  /**
    * Lambda to center then scale an input Seq[Double] so that output Seq has mean 0 and standard deviation 1.
    * @param zeroPrecision a threshold below which a value is considered 0
    * @return a lambda for z normalization with given precision
    */
  def zNormalize(zeroPrecision: Double) = (seq: Seq[Double]) => {

    val mean = seq.sum / seq.size
    val add: (Double, Double) => Double = (acc, elem) => acc + Math.pow(elem - mean, 2D)
    val std = Math.sqrt(seq.foldLeft(0D)(add) / seq.size)
    if ((std - 0D).abs < zeroPrecision) seq.map(x => 0D)
    else seq.map( x => (x - mean) / std )
  }

  /**
    * Lambda to compute categorical binary encoding of a category index scalar value.
    * For example, a <categoryIndex> of 2 will be encoded as 001 followed by a number of zeros so that the Seq is of size <categoriesNumber>
    * @param categoriesNumber the number of categories, which will be the size of the output Seq[Double]
    * @return a lambda for categorical binary encoding of size <categoriesNumber>
    */
  def categoricalBinaryEncoding(categoriesNumber: Int) = ((categoryIndex: Double) => {
    Seq.fill(categoriesNumber)(0D).updated(Math.floor(categoryIndex).toInt, 1D)
  })

  /**
    * Transforms a Spark ML Vector into a Seq
    */
  val vecToSeq = ((vec: org.apache.spark.ml.linalg.Vector) => {
    vec.toArray.toSeq
  })

  /**
    * Transforms a Seq into a Spark ML Vector
    */
  val seqToVec = ((seq: Seq[Double]) => {
    Vectors.dense(seq.toArray)
  })

}