package com.github.unsupervise.spark.tss.steps

/**
  * Created by antho on 26/09/2019.
  */
import com.github.unsupervise.spark.tss.core._
import org.apache.spark.sql.functions.{min, max, col, array_max, array_min, mean}

/**
  * Lazy Interpolated Fourier Vectorization Step Command.
  * This Step transforms an input TSS series into its aligned accross all series and interpolated Fourier periodogram vectorization.
  * @param interpolatedPointsNumber the maximal number of interpolated points number. The actual number can be reduced if some series are not long enough.
  * @param inSeriesColName the input series column name from which computing the vectorization
  * @param inTimeGranularityColName the input time granularity column name
  * @param outZSeriesColName the output column name for the z-normalized version of the series located in <inSeriesColName> column
  * @param outFourierColName the output column name for the Discrete Fourier Transform of the z-normalized series located in <outZSeriesColName> column
  * @param outFourierFrequenciesColName the output column name for the Discrete Fourier Transform frequency values of the z-normalized series located in <outZSeriesColName> column
  * @param outPeriodogramColName the output column name for the Discrete Fourier Transform Periodogram of the z-normalized series located in <outZSeriesColName> column
  * @param outInterpolatedFourierFrequenciesColName the output column name for the interpolated Discrete Fourier Transform frequency values of the z-normalized series located in <outZSeriesColName> column, after aligning each series Fourier Transform frequency bases
  * @param outInterpolatedPeriodogramColName the output column name for the interpolated Discrete Fourier Transform of the z-normalized series located in <outZSeriesColName> column, after aligning each series Fourier Transform frequency bases
  * @param outReconstructedZSeriesColName  the output column name for the reconstructed z-normalized series from the Discrete Fourier Transform (check purposes)
  * @param precision the precision for the z-normalisation, value below which 0 is enforced.
  */
case class InterpolatedFourierVectorization(interpolatedPointsNumber: Int,
                                            inSeriesColName: String = TSS.SERIES_COLNAME, inTimeGranularityColName: String = TSS.TIMEGRANULARITY_COLNAME,
                                            outZSeriesColName: String = "zseries", outFourierColName: String = "fourier",
                                            outFourierFrequenciesColName: String = "fourierFrequencies",
                                            outPeriodogramColName: String = "fourierPeriodogram", outInterpolatedFourierFrequenciesColName: String = "fourierPeriodogramInterpolationFrequencies",
                                            outInterpolatedPeriodogramColName: String = "fourierPeriodogramLinearInterpolation",
                                            outReconstructedZSeriesColName: String = "reconstructedZSeries",
                                            precision: Double = 0.0000000000001) extends Step {

  override def apply(in: TSS): TSS = {
    val withFourierTable =
      in.addZNormalized(outZSeriesColName, inSeriesColName, precision)
        .addDFT(outFourierColName, outZSeriesColName)
        .addDFTFrequencies(outFourierFrequenciesColName, inSeriesColName, inTimeGranularityColName)
        .addDFTPeriodogram(outPeriodogramColName, outFourierColName)
        .addInverseDFT(outReconstructedZSeriesColName, outFourierColName).persist
    val meanFourierFrequencyStep = withFourierTable.colSeqFirstStep(outFourierFrequenciesColName).agg(mean("value")).first.getDouble(0)
    //Compute linear interpolation of periodograms on wanted number of first points
    //over a mean scenario frequency step basis. Remove 0 bin Fourier coefficient to avoid 0 columns in matrices afterwards
    //Restrict to smallest interpolation points space if some series have frequencies outside the original bounds
    //=> Work on the intersection of frequency spaces
    val newInterpolationSamplePoints = Array.range(1, interpolatedPointsNumber).map(_.toDouble * meanFourierFrequencyStep)
    val minMaxAndMaxMinFourierFrequency = withFourierTable.series.select(min(array_max(col(outFourierFrequenciesColName))), max(array_min(col(outFourierFrequenciesColName)))).first
    val minMaxFourierFrequency = minMaxAndMaxMinFourierFrequency.getDouble(0)
    val maxMinFourierFrequency = minMaxAndMaxMinFourierFrequency.getDouble(1)
    val keptInterpolationSamplePoints = newInterpolationSamplePoints.filter(x => x < minMaxFourierFrequency && x > maxMinFourierFrequency)
    withFourierTable
      .addConstant(outInterpolatedFourierFrequenciesColName, keptInterpolationSamplePoints)
      .addLinearInterpolationPoints(outInterpolatedPeriodogramColName, outFourierFrequenciesColName, outPeriodogramColName, keptInterpolationSamplePoints)
  }

}
