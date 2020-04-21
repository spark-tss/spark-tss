package com.github.acout.spark.tss.core

import com.github.acout.spark.tss.functions._
import java.io.{BufferedWriter, File, FileWriter}
import java.util.UUID

import breeze.interpolation._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.signal.fourierFreq
import breeze.stats.distributions.MultivariateGaussian
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.clustering._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.linalg.EigenValueDecomposition
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.{UserDefinedAggregateFunction, UserDefinedFunction}
import org.apache.spark.sql.functions.{first, sum, _}
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.clustering4ever.clustering.indices.InternalIndicesDistributed
import org.clustering4ever.clusterizables.EasyClusterizable
import org.clustering4ever.math.distances.scalar.Euclidean
import org.clustering4ever.scala.clustering.tensor.EigenValue
import org.clustering4ever.vectors.{GVector, ScalarVector}
import smile.{mds, projection}
import xyz.florentforest.spark.ml.som.SOM
import smile.clustering
import spire.ClassTag
// import com.github.yazidjanati.BayesianTuner
import org.apache.hadoop.fs.{FileSystem, Path}

/**
  * TSS class stands for Time Series Set and is the base class for row time series storage and analysis.
  * It is a wrapper around a DataFrame object evolving accross TS parallel pipelines
  * @param inSeries the DataFrame containing time series data and diverse representations / transformations
  * @param forceIds whether to force the recomputation of unique ids for each row or not (costly phase, not needed after simple mappings for example)
  */
case class TSS(inSeries: DataFrame, forceIds: Boolean = false) {

  //Auto add a unique id column if it does not exist or if ids forcing is required
  val series =
    if(inSeries.columns.contains(TSS.ID_COLNAME) && !forceIds) inSeries
    else inSeries.withColumn(TSS.ID_COLNAME, monotonically_increasing_id())

  /**
    * The lazily evaluated rows count, costly to recompute each time.
    */
  lazy val count = series.count()

  import series.sparkSession.implicits._
  //import series.sparkSession.sqlContext.implicits._

  //USER DEFINED FUNCTIONS (UDF)
  /**
    * UDF for computing Fast Fourier Transform on a Seq[Double] column
    */
  val dftUDF = udf((ts: Seq[Double]) => fft(ts))
  /**
    * UDF for computing Inverse Fast Fourier Transform on a Seq[Double] column
    */
  val inverseDFTUDF = udf((ts: Seq[Double]) => inverseFFT(ts))
  /**
    * UDF for computing Fourier periodogram from a DFT Seq[Double] column
    */
  val dftPeriodogramUDF = udf((dft: Seq[Double]) => dftPeriodogram(dft))
  /**
    * UDF for computing Fourier frequencies from a Seq[Double] column and a time granularity column
    */
  val dftFrequenciesUDF = udf((ts: Seq[Double], dt: Double) => {
    fourierFreq(ts.size, dt = dt).toArray
  })

  /**
    * UDF for computing Discrete Wavelet Transform from a Seq[Double] column
    * @param filter the wavelet family name, as supported by SMILE library
    */
  def dwtUDF(filter: String) = udf((ts: Seq[Double]) => {
    dwt(filter)(ts)
  })
  /**
    * UDF for computing Inverse Discrete Wavelet Transform from a Seq[Double] column
    * @param filter the wavelet family name, as supported by SMILE library
    */
  def idwtUDF(filter: String) = udf((ts: Seq[Double]) => {
    idwt(filter)(ts)
  })

  /**
    * UDF generator for one pass computation of DFT => DFT Periodogram => Log10 shrinkage and avoid some Spark overhead
    * @param safetyOffset the ofsset to add to log10 to avoid log10(0) occuring
    * @return the UDF function to apply to a Seq[Double] column representing a discrete time series
    */
  //Method for combining DFT, Periodogram computation then Log10 shrinkage, for having one transformation step
  def log10DFTPeriodogramUDF(safetyOffset: Double) = udf((ts: Seq[Double]) => {
    dftPeriodogram(fft(ts)).map(x => scala.math.log10(x + safetyOffset))
  })

  /**
    * UDF for concatenating decorators Map and warnings list as a Seq[String],
    * to join corresponding plan files together.
    */
  val concatMapWarningsUDF = udf((m: Map[String, String], l: Seq[String]) => {
    m ++ l.filter(_.split("::").length > 2).map(x => {
      val splittedRow = x.split("::")
      splittedRow(0) + "::" + splittedRow(1) -> splittedRow(2)
    }).toMap
  })
  /**
    * UDF for centering a Seq[Double] column, thus leading to 0 mean for its values
    */
  val centerUDF = udf((seq: Seq[Double]) => {center(seq)})

  /**
    * UDF generator for transforming a Seq[Double] column into its weighted version
    * @param weights the weights vector to apply in an element wize fashion
    * @return the UDF function to apply to a Seq[Double] for the given weights
    */
  def weightedUDF(weights: Seq[Double]) = udf((seq: Seq[Double]) => {weighted(seq, weights)})
  /**
    * UDF for concatenating decorators Map with Metadata as String to join corresponding
    * plan files in the adequate manner.
    */
  val concatMapMetadataUDF = udf((m: Map[String, String], s: String) => {
    //TODO: Remove dependency from "##" string
    if(s != null){
      m ++ s.split("##").map(x => {
        val splitted = x.split("::")
        splitted(0) -> splitted(1)
      })
    }else{
      m
    }
  })
  /**
    * UDF to compute euclidean distance between two Seq[Double] columns of same size
    */
  def euclideanDistanceUDF(squared: Boolean = true) = udf((seq1: Seq[Double], seq2: Seq[Double]) => {
    assert(seq1.size == seq2.size)
    val res: Double = TSS.euclideanF(squared)(seq1, seq2)
    res
  })
  /**
    * UDF generator to compute cubic interpolation for some xs
    * given a discrete time series described by xs and ys Seq[Double] columns.
    * This variant is tailed for per row cubic model fitting, but constant xs queried accros rows
    * @param newXs the collection of xs values for which interpolated values are requested, after fitting the model
    * @return an UDF for the constant query xs values
    */
  def cubicInterpolationPointsUDF(newXs: Seq[Double]) = udf((xs: Seq[Double], ys: Seq[Double]) => {
    //TODO: avoid computing splines on points non necessary for newXs sampling
    val t = System.nanoTime()
    val vnxs = new DenseVector(newXs.toArray)
    val t2 = System.nanoTime()
    val maxNewX = newXs.max
    val lastUsefulX = xs.find(x => x > maxNewX).get
    val usefulXs = xs.filter(x => x <= lastUsefulX)
    val usefulYs = ys.take(usefulXs.size)
    val vuxs = new DenseVector(usefulXs.toArray)
    val vuys = new DenseVector(usefulYs.toArray)
    val f = new CubicInterpolator(vuxs, vuys)
    val t3 = System.nanoTime()
    val res = f(vnxs).toArray
    val t4 = System.nanoTime()
    println(List(t, t2, t3, t4).sliding(2).map(x => x(1) - x(0)).toArray.mkString(","))
    res
  })
  /**
    * UDF to compute cubic interpolation for some xs
    * given a discrete time series described by xs and ys Seq[Double] columns.
    * This variant is tailed for per row cubic model fitting and per row model query of unknown xs
    */
  def cubicInterpolationPointsUDF = udf((xs: Seq[Double], ys: Seq[Double], newXs: Seq[Double]) => {
    println(xs.size + " " + ys.size)
    val maxNewX = newXs.max
    val lastUsefulX = xs.find(x => x > maxNewX).get
    val usefulXs = xs.filter(x => x <= lastUsefulX)
    val usefulYs = ys.take(usefulXs.size)
    val vuxs = new DenseVector(usefulXs.toArray)
    val vuys = new DenseVector(usefulYs.toArray)
    val f = new CubicInterpolator(vuxs, vuys)
    f(new DenseVector(newXs.toArray)).toArray
  })
  /**
    * UDF generator to compute linear interpolation for some xs
    * given a discrete time series described by xs and ys Seq[Double] columns.
    * This variant is tailed for per row linear model fitting, but constant xs queried accros rows
    * @param newXs the collection of xs values for which interpolated values are requested, after fitting the model
    * @return an UDF for the constant query xs values
    */
  def linearInterpolationPointsUDF(newXs: Seq[Double]) = udf((xs: Seq[Double], ys: Seq[Double]) => {
    val vxs = new DenseVector(xs.toArray)
    val vys = new DenseVector(ys.toArray)
    val vnxs = new DenseVector(newXs.toArray)
    val f = new LinearInterpolator(vxs, vys)
    val res = f(vnxs).toArray
    res
  })
  /**
    * UDF to compute linear interpolation for some xs
    * given a discrete time series described by xs and ys Seq[Double] columns.
    * This variant is tailed for per row linear model fitting and per row model query of unknown xs
    */
  def linearInterpolationPointsUDF = udf((xs: Seq[Double], ys: Seq[Double], newXs: Seq[Double]) => {
    val f = new LinearInterpolator(new DenseVector(xs.toArray), new DenseVector(ys.toArray))
    f(new DenseVector(newXs.toArray)).toArray
  })
  /**
    * UDF to generate scatterplot items of time series on EvilPlot format, given time range,
    * and values to plot at the corresponding xs + decorators to get more meaningful labels on plot.
    */
  def buildShowAllPointsUDF = {
    udf(
      (timeFrom: Double, timeTo: Double, timeGranularity: Double, values: Seq[Double], decorators: Map[String, Any]) => {
        val times = timeFrom to timeTo by timeGranularity
        val label = decorators.map(x => x._1 + ":" + x._2.toString).reduce(_ + "|" + _)
        ???
        // Map("label" -> label, "points" -> times.zip(values).map(x => Point(x._1, x._2)))
        ???
      }
    )
    ???
  }
  /**
    * UDF generator for low pass filtering on a Seq[Double] signal, such as DFT result
    * @param keepQuantity the number of lowest frequencies to keep in the filtered signal
    * @return an UDF for the given keepQuantity value
    */
  def symmetricLowPassUDF(keepQuantity: Int) = udf((sequence: Seq[Double]) => {
    val len = sequence.size
    sequence.indices.map(i => if(i < keepQuantity || i > len - keepQuantity) sequence(i) else 0)
  })
  /**
    * UDF generator to set some values, which set is described by a predicate function from indices, to 0
    * @param filterFun the predicate function which returns true iff the input Seq[Double] corresponding index must be set to 0
    * @return an UDF for the corresponding predicate function
    */
  def nullifyValuesUDF(filterFun: Int => Boolean) = udf((seq: Seq[Double]) => {
    seq.indices.map(i => if(filterFun(i)) 0D else seq(i))
  })
  /**
    * UDF generator to set some values, which set is described by a indices slice
    * @param startIndex the series index where to begin 0 setting
    * @param quantity the number of series elements to set to 0 from <startIndex>
    * @return an UDF for the corresponding slice
    */
  def nullifyValuesSliceUDF(startIndex: Int, quantity: Int) = udf((seq: Seq[Double]) => {
    seq.indices.map(i => if(i >= startIndex && i < startIndex + quantity) 0D else seq(i))
  })
  /**
    * An UDF generator for scaling a time series so that their value standard deviation is 1
    * @param zeroPrecision the threshold below which a value is considered 0
    * @return an UDF for the given precision
    */
  def scaleUDF(zeroPrecision: Double) = udf((seq: Seq[Double]) => {scale(zeroPrecision)(seq)})
  /**
    * An UDF to normalize a Seq[Double] by its sum, so that the new values sum to 1
    */
  val normalizeSumUDF = udf((seq: Seq[Double]) => {normalizeSum(seq)})
  /**
    * An UDF generator to Z-Normalize a Seq[Double] (centering + scaling it by row)
    * @param zeroPrecision the threshold below which a value is considered 0
    * @return an UDF for the given precision
    */
  def zNormalizeUDF(zeroPrecision: Double) = udf((seq: Seq[Double]) => {zNormalize(zeroPrecision)(seq)})
  /**
    * A UDF generator to compute element-wize log10 on a Seq[Double]
    * @param safetyOffset the offset to add to input values to avoid log10(0) error
    * @return an UDF for given offset
    */
  def log10UDF(safetyOffset: Double = 0D) = udf((seq: Seq[Double]) => {com.github.acout.spark.tss.functions.log10(safetyOffset)(seq)})
  /**
    * A UDF generator to compute categorical binary encoding of a category index scalar value.
    * For example, a value of 2 will be encoded as 001 followed by a number of zeros so that the Seq is of size <categoriesNumber>
    * @param categoriesNumber the number of categories, which will be the size of the output Seq[Double]
    * @return an UDF for given categories number
    */
  def categoricalBinaryEncodingUDF(categoriesNumber: Int) = udf((category: Double) => categoricalBinaryEncoding(categoriesNumber)(category))
  //END USER DEFINED FUNCTIONS

  //PRIVATE FUNCTIONS
  //Escape the column name for Spark understanding
  private[this] def esc(colName: String) = ("`" + colName + "`").replace("``", "`")
  //Remove Spark escaping
  private[this] def unesc(colName: String) = colName.replace("`", "")
  //Helper to generate EvilPlot formatted time series
  private[this] def rowToLabeledPoint(ts: Row, timeFromId: Int, timeToId: Int, timeGranularityId: Int, decoratorsId: Int, valuesId: Int) = {
    val times = (ts.getDouble(timeFromId) to ts.getDouble(timeToId) by ts.getDouble(timeGranularityId))
    val values = ts.getSeq[Double](valuesId)
    val valuesColName = ts.schema(valuesId).name
    val label = ts.getMap[String, String](decoratorsId).updated("colName", valuesColName)
      .map(x => x._1 + ":" + x._2.toString).reduce(_ + "|" + _)
    // (label, times.zip(values).map(x => Point(x._1, x._2)))
    ???
  }
  //Retrieve values of single decorator for every row as an RDD
  private[this] def getDecoratorDomainRDD(decoratorName: String) = {
    series.select(TSS.DECORATORS_COLNAME).rdd.map(_.getMap[String, String](0)(decoratorName))
  }
  //Group by a set of decorators
  private[this] def groupByDecorators(decoratorNames: String*) = {
    series.groupByKey(x => {
      val decorators = x.getMap[String, String](x.fieldIndex(TSS.DECORATORS_COLNAME))
      decoratorNames.map(n => decorators(n))
    })
  }
  //END PRIVATE FUNCTIONS

  //INTERFACE FUNCTIONS
  /**
    * Adds a column to TSS containing the DFT of another one.
    * @param outColName the new column name
    * @param sourceColName the source column name, to compute DFT on
    * @return a TSS with the extra column creation planned
    */
  def addDFT(outColName: String, sourceColName: String = TSS.SERIES_COLNAME) = {
    addUDFColumn(outColName, sourceColName, dftUDF)
  }
  /**
    * Adds a column to TSS containing the Inverse DFT of another one.
    * @param outColName the new column name
    * @param sourceColName the source column name, to compute Inverse DFT on
    * @return a TSS with the extra column creation planned
    */
  def addInverseDFT(outColName: String, sourceColName: String) = {
    addUDFColumn(outColName, sourceColName, inverseDFTUDF)
  }
  /**
    * Adds a column to TSS containing the Fourier Periodogram of another DFT one.
    * @param outColName the new column name
    * @param sourceColName the source column name, to compute Fourier Periodogram on
    * @return a TSS with the extra column creation planned
    */
  def addDFTPeriodogram(outColName: String, sourceColName: String) = {
    addUDFColumn(outColName, sourceColName, dftPeriodogramUDF)
  }
  /**
    * Adds a column to TSS containing the DFT Frequencies of another one.
    * @param outColName the new column name
    * @param sourceColName the source column name, to compute DFT Frequencies on
    * @param granularityColName the delta t value column name, used for frequency computation
    * @return a TSS with the extra column creation planned
    */
  def addDFTFrequencies(outColName: String, sourceColName: String, granularityColName: String) = {
    addUDFColumn(outColName, Array(sourceColName, granularityColName), dftFrequenciesUDF)
  }
  /**
    * Adds a column to compute the whole DFT => Periodogram => Log10 sequence on one pass
    * @param outColName the new column name
    * @param sourceColName the source column name, to compute all treatments on (typically, a time series)
    * @param safetyOffset the offset to add to each value before applying log10, to avoid log10(0) errors
    * @return a TSS with the extra column creation planned
    */
  def addLog10DFTPeriodogram(outColName: String, sourceColName: String, safetyOffset: Double = 0) = {
    addUDFColumn(outColName, sourceColName, log10DFTPeriodogramUDF(safetyOffset))
  }
  /**
    * Adds a column to TSS containing the DWT of another one.
    * @param outColName the new column name
    * @param sourceColName the source column name, to compute DWT on
    * @return a TSS with the extra column creation planned
    */
  def addDWTSmile(outColName: String, sourceColName: String, filter: String) = {
    addUDFColumn(outColName, sourceColName, dwtUDF(filter))
  }
  /**
    * Adds a column to TSS containing the Inverse DWT of another one.
    * @param outColName the new column name
    * @param sourceColName the source column name, to compute Inverse DWT on
    * @return a TSS with the extra column creation planned
    */
  def addInverseDWTSmile(outColName: String, sourceColName: String, filter: String) = {
    addUDFColumn(outColName, sourceColName, idwtUDF(filter))
  }
  /**
    * Adds a column to compute the per row euclidean distance between two columns
    * @param outColName the new column name
    * @param sourceColName1 the first Seq[Double] of size k column name to compute distance on
    * @param sourceColName2 the second Seq[Double] of size k column name to compute distance on
    * @param squared whether to retrieve the squared Euclidean distance or not
    * @return a TSS with the extra column creation planned
    */
  def addSeqEuclideanDistance(outColName: String, sourceColName1: String, sourceColName2: String, squared: Boolean = true) = {
    addUDFColumn(outColName, Array(sourceColName1, sourceColName2), euclideanDistanceUDF(squared))
  }
  /**
    * Adds a column to compute the per row relative ratio of a column value (called deviant or d)
    * with respect to another one (called the referent or r).
    * More precisely, the value v in the new column will be (d - r) / r for each row.
    * @param outColName the new column name
    * @param deviantColName the column which scalar value represents the one for which the relative ratio is computed
    * @param referentColName the column which scalar value represents the reference value.
    * @return a TSS with the extra column creation planned
    */
  def addRelativeRatio(outColName: String, deviantColName: String, referentColName: String) = {
    val deviantCol = series(esc(deviantColName))
    val referentCol = series(esc(referentColName))
    TSS(series.withColumn(outColName, deviantCol.minus(referentCol).divide(referentCol)))
  }
  /**
    * Adds a column with low pass filtering of an input Seq[Double] DFT signal column
    * @param outColName the new column name
    * @param sourceColName the column containing the DFT Seq[Double] to which applying the low pass filter
    * @param keepQuantity the number of low frequency values to keep in the filtered signal
    * @return a TSS with the extra column creation planned
    */
  def addSymmetricLowPassFilter(outColName: String, sourceColName: String, keepQuantity: Int) = {
    addUDFColumn(outColName, sourceColName, symmetricLowPassUDF(keepQuantity))
    //com.github.acout.spark.tss.core.TSS(series.withColumn(outColName, symmetricLowPassUDF(keepQuantity)(series(sourceColName))))
  }
  /**
    * Adds a Seq[Double] column with interpolations of a constant set of xs accross all rows from
    * per row cubic model of points given by Seq[Double] pair of columns.
    * @param outColName the added column name
    * @param xColName the column name containing the xs for the cubic model fitting
    * @param yColName the column name containing the ys for the cubic model fitting
    * @param newXs the constant Seq[Double] containing the points to evaluate on cubic models for each row
    * @return a TSS with the extra column creation planned
    */
  def addCubicInterpolationPoints(outColName: String, xColName: String, yColName: String, newXs: Seq[Double]) = {
    addUDFColumn(outColName, Array(xColName, yColName), cubicInterpolationPointsUDF(newXs))
  }
  /**
    * Adds a Seq[Double] column with interpolations of a per row set of xs from per row cubic model of points given
    * by Seq[Double] pair of columns.
    * @param outColName the added column name
    * @param xColName the column name containing the xs for the linear model fitting
    * @param yColName the column name containing the ys for the linear model fitting
    * @param newXsColName the column name containing the points to evaluate on same row cubic model
    * @return a TSS with the extra column creation planned
    */
  def addCubicInterpolationPoints(outColName: String, xColName: String, yColName: String, newXsColName: String) = {
    addUDFColumn(outColName, Array(xColName, yColName, newXsColName), cubicInterpolationPointsUDF)
  }
  /**
    * Adds a Seq[Double] column with interpolations of a constant set of xs accross all rows from
    * per row linear model of points given by Seq[Double] pair of columns.
    * @param outColName the added column name
    * @param xColName the column name containing the xs for the linear model fitting
    * @param yColName the column name containing the ys for the linear model fitting
    * @param newXs the constant Seq[Double] containing the points to evaluate on linear models for each row
    * @return a TSS with the extra column creation planned
    */
  def addLinearInterpolationPoints(outColName: String, xColName: String, yColName: String, newXs: Seq[Double]) = {
    addUDFColumn(outColName, Array(xColName, yColName), linearInterpolationPointsUDF(newXs))
  }
  /**
    * Adds a Seq[Double] column with interpolations of a per row set of xs from per row linear model of points given
    * by Seq[Double] pair of columns.
    * @param outColName the added column name
    * @param xColName the column name containing the xs for the cubic model fitting
    * @param yColName the column name containing the ys for the cubic model fitting
    * @param newXsColName the column name containing the points to evaluate on same row cubic model
    * @return a TSS with the extra column creation planned
    */
  def addLinearInterpolationPoints(outColName: String, xColName: String, yColName: String, newXsColName: String) = {
    addUDFColumn(outColName, Array(xColName, yColName, newXsColName), linearInterpolationPointsUDF)
  }

  /**
    * Adds a Seq[Double] column with element-wize log10 values of a source Seq[Double] column in same order.
    * @param outColName the added column name
    * @param sourceColName the column name from which computing the element-wize log10
    * @param safetyOffset the offset to add to every value prior log10 call, to avoid log10(0) errors
    * @return a TSS with the extra column creation planned
    */
  def addLog10(outColName: String, sourceColName: String, safetyOffset: Double = 0) = {
    addUDFColumn(outColName, sourceColName, log10UDF(safetyOffset))
  }

  /**
    * Adds a Seq[Double] column containing the scaled transformation of an input Seq[Double] column,
    * so that standard deviation of the new column elements is equal to 1.
    * @param outColName the added column name
    * @param sourceColName the column name from which computing the scale transformation
    * @param zeroPrecision the threshold below which a value is considered 0
    * @return a TSS with the extra column creation planned
    */
  def addScaled(outColName: String, sourceColName: String, zeroPrecision: Double) = {
    addUDFColumn(outColName, sourceColName, scaleUDF(zeroPrecision))
  }
  /**
    * Adds a Seq[Double] column containing the centered transformation of an input Seq[Double] column,
    * so that mean of the new column elements is equal to 0.
    * @param outColName the added column name
    * @param sourceColName the column name from which computing the transformation
    * @return a TSS with the extra column creation planned
    */
  def addCentered(outColName: String, sourceColName: String) = {
    addUDFColumn(outColName, sourceColName, centerUDF)
  }

  /**
    * Adds a Seq[Double] column containing the element-wise weighted transformation of an input Seq[Double] column.
    * @param outColName the added column name
    * @param sourceColName the column name from which computing the transformation
    * @param weights the weights vector
    * @return a TSS with the extra column creation planned
    */
  def addWeighted(outColName: String, sourceColName: String, weights: Seq[Double]) = {
    addUDFColumn(outColName, sourceColName, weightedUDF(weights))
  }

  /**
    * Adds a Seq[Double] column containing the centered then scaled transformation of an input Seq[Double] column,
    * so that mean of the new column elements is equal to 0 and standard deviation is equal to 1.
    * @param outColName the added column name
    * @param sourceColName the column name from which computing the transformations
    * @param zeroPrecision the threshold below which a value is considered 0
    * @return a TSS with the extra column creation planned
    */
  def addZNormalized(outColName: String, sourceColName: String, zeroPrecision: Double) = {
    addUDFColumn(outColName, sourceColName, zNormalizeUDF(zeroPrecision))
  }
  /**
    * Adds a Seq[Double] column containing the normalized by sum transformation of an input Seq[Double] column,
    * so that sum of the new column elements is equal to 1.
    * @param outColName the added column name
    * @param sourceColName the column name from which computing the transformations
    * @return a TSS with the extra column creation planned
    */
  def addNormalizeSum(outColName: String, sourceColName: String) = {
    addUDFColumn(outColName, sourceColName, normalizeSumUDF)
  }

  /**
    * Adds a Spark ML Vector type column from an input Seq[Double] column,
    * making the new column compatible with Spark ML processings.
    * @param outColName the added column name
    * @param sourceColName the Seq[Double] column name to transform
    * @return a TSS with the extra column creation planned
    */
  def addMLVectorized(outColName: String, sourceColName: String) = {
    addUDFColumn(outColName, sourceColName, udf(seqToVec))
  }

  /**
    * Adds a Seq[Double] type column from an input Spark ML Vector column.
    * @param outColName the added column name
    * @param sourceColName the Spark ML Vector column name to transform
    * @return a TSS with the extra column creation planned
    */
  def addSeqFromMLVector(outColName: String, sourceColName: String) = {
    addUDFColumn(outColName, sourceColName, udf(vecToSeq))
  }

  /**
    * Adds a Spark ML Vector type column from an array of input columns
    * @param outColName the assembled vector output column name
    * @param sourceColNames the input column names (scalar or sequence) to concatenate and flatten
    * @return a TSS with the extra column creation planned
    */
  def addMLVectorAssembly(outColName: String, sourceColNames: Array[String]) = {
    val assembler = new VectorAssembler()
      .setInputCols(sourceColNames)
      .setOutputCol(outColName)
    TSS(assembler.transform(series))
  }

  /**
    * Adds a scalar column containing the column-wize scaled and/or centered transformation of input scalar column
    * @param outColName the added column name
    * @param inColName the input column with scalars to scale so that the new column has standard deviation of 1 and
    *                  or mean of 0
    * @param scale whether to scale the input column so that output column has standard deviation 1
    * @param center whether to center the input column so that output column has mean 0
    * @return a TSS with the extra column creation planned
    */
  def addColScaled(outColName: String, inColName: String, scale: Boolean = true, center: Boolean = false) = {
    val scaler = new StandardScaler()
      .setInputCol(inColName)
      .setOutputCol(outColName)
      .setWithStd(scale)
      .setWithMean(center)
    val scalerModel = scaler.fit(series)
    TSS(scalerModel.transform(series))
  }

  /**
    * Learn a Spark ML Decision Tree Classifier and store predictions in TSS columns + model information & evaluations in output file.
    * @param outPredictColName the ouput column name where to store hard class assignments of decision tree
    * @param outProbasColName the output column name where to store soft class assignments of decision tree
    * @param sourceFeaturesColName the input Vector column name containing individual features to use for classification
    * @param sourceLabelColName the input scalar column name containing the individual labels to predict
    * @param trainingSetSizeRatio a Double in [0; 1] indicating the part of DataFrame used for training, the leftovers being assigned to test set
    * @param impurityFunName an impurity function name, as supported by Spark ML
    * @param maxBinsNumber the maximum number of tree leaves allowed during learning
    * @param maxTreeDepth the maximal tree depth allowed during learning
    * @param evaluators an Array of evaluator names, as supported by Spark ML
    * @param modelOutFilePath the file path to store the model information in Spark ML format, if requested
    * @param evaluationsOutFile an output File object where to store the evaluations Vector, if requested
    * @return a TSS with the extra prediciton columns creation planned
    */
  def addMLDecisionTree(outPredictColName: String, outProbasColName: String, sourceFeaturesColName: String, sourceLabelColName: String,
                        trainingSetSizeRatio: Double, impurityFunName: String, maxBinsNumber: Int, maxTreeDepth: Int, evaluators: Array[String],
                        modelOutFilePath: Option[String] = None, evaluationsOutFile: Option[File] = None) = {
    series.take(1)
    val indexedLabelColName = "indexed_" + sourceLabelColName
    val indexedPredictionColName = "indexed_" + outPredictColName
    val labelIndexer = new StringIndexer()
      .setInputCol(sourceLabelColName)
      .setOutputCol(indexedLabelColName)
      .fit(series)
    val Array(trainingData, testData) = series.randomSplit(Array(trainingSetSizeRatio, 1D - trainingSetSizeRatio))
    val dt = new DecisionTreeClassifier()
      .setLabelCol(indexedLabelColName)
      .setFeaturesCol(sourceFeaturesColName)
      .setImpurity(impurityFunName)
      .setMaxBins(maxBinsNumber)
      .setMaxDepth(maxTreeDepth)
      .setPredictionCol(indexedPredictionColName)
    val model = dt.fit(trainingData)
    val predictions = model.transform(testData)
    val evaluations = evaluators.map(e => {
      new MulticlassClassificationEvaluator()
        .setLabelCol(indexedLabelColName)
        .setPredictionCol(indexedPredictionColName)
        .setMetricName(e).evaluate(predictions)
    })
    println("evaluations = " + evaluations)
    modelOutFilePath.map(fp => {
      model.save(fp)
    })
    val labelConverter = new IndexToString()
      .setInputCol(indexedPredictionColName)
      .setOutputCol(outPredictColName)
      .setLabels(labelIndexer.labels)
    TSS(labelConverter.transform(model.transform(series)).drop(indexedLabelColName, indexedPredictionColName))
  }

  /**
    * Compute inter-rows distance as a TSDS object
    * @param colName the column name containing the Seq[Double] features set used for distances computation
    * @param dist the binary distance function from Seq[Double]² to Double
    * @param symmetric whether the distance is symmetric, in which case computations are only performed once per pair
    * @return a TSDS object with the distances planned to process
    */
  def rowDistances[A](colName: String, dist: (Seq[A], Seq[A]) => Double, symmetric: Boolean = true) = {
    val subSeries = series.select(TSS.ID_COLNAME, colName).persist()
    subSeries.take(1) //Ensure persist is achieved before join is called
    //val oldConf = subSeries.sparkSession.conf.get("spark.sql.autoBroadcastJoinThreshold")
    //subSeries.sparkSession.conf.set("spark.sql.autoBroadcastJoinThreshold", 0)
    val otherSubSeries =
      subSeries.select(subSeries(TSS.ID_COLNAME).alias(TSS.ID_COLNAME + "_2"),
                       subSeries(colName).alias(colName + "_2"))
    val crossSeries = subSeries.crossJoin(otherSubSeries)
    val ds = crossSeries.flatMap(x => {
        val id1 = x.getLong(0)
        val id2 = x.getLong(2)
        if((symmetric && id1 <= id2) || !symmetric){
          Some(IndexedDistance(id1, id2, dist(x.getSeq[A](1), x.getSeq[A](3))))
        }else{
          None
        }
      })
    val fullDS = if(symmetric){
      val symDS = ds.select(ds("id2").alias("id1"), ds("id1").alias("id2"), ds("distance")).as[IndexedDistance]
      //Remove ego distances, since already added by previous step
      ds.union(symDS.where(symDS("id1") =!= symDS("id2")))
    }else ds
    subSeries.unpersist()
    //subSeries.sparkSession.conf.set("spark.sql.autoBroadcastJoinThreshold", oldConf)
    TSDS(fullDS, count.toInt)
  }

  /**
    * Compute rows to specific points distance as a TSDS object. Useful to compute individuals to clustering prototypes distances for example.
    * @param sourceColName the column name containing the Seq[Double] features set of size k used for left hand side of distances computation
    * @param targetPoints the constant Dataset of target points of dimension k to use as right hand size of distances computation
    * @return a TSDS object with the distances planned to process
    */
  def rowDistances[A](sourceColName: String, targetPoints: Dataset[(Long, Seq[A])], dist: (Seq[A], Seq[A]) => Double) = {
    targetPoints.persist()
    targetPoints.take(1) //Ensure persist is achieved before join is called
    val crossSeries = series.select(TSS.ID_COLNAME, sourceColName).crossJoin(targetPoints)
    val ds = crossSeries.map(x => IndexedDistance(x.getLong(0), x.getLong(2), dist(x.getSeq[A](1), x.getSeq[A](3))))
    targetPoints.unpersist()
    TSDS(ds, count.toInt)
  }

  /**
    * Get Closest Points to Given Target Points, one for each target, as a 3 columns DataFrame
    * @param sourceColName the name of the column containing the vector coordinates to compare to target points coordinates
    * @param targetPoints the target point coordinates
    * @param dist the distance function used to compute individual to target distances, using coordinates defined by <sourceColName> and <targetPoints>
    * @tparam A the type of <sourceColName> vector representation scalars
    * @return a DataFrame with 3 columns:
    *         - closestPoint containing the id of the closest points for a given target point,
    *         - <TSDS.ID2_COLNAME> defining the id of target point the closest point refers to,
    *         - <TSDS.DISTANCE_COLNAME> giving the distance between a target point and its closest point.
    */
  def getClosestPointsToTargets[A](sourceColName: String, targetPoints: Dataset[(Long, Seq[A])], dist: (Seq[A], Seq[A]) => Double): DataFrame = {
    val dists = rowDistances(sourceColName, targetPoints, dist)
    dists.getAggregateById(first(dists.indexedDistances(TSDS.ID1_COLNAME)).alias("closestPoint"), TSDS.ID2_COLNAME, TSDS.DISTANCE_COLNAME)
  }

  /**
    * Transforms a scalar value representing a category value in a given domain, by a binary sequence with a 1 at the
    * only index corresponding to the original value
    * @param outColName the output Seq[Double] column name, where each value is binary
    * @param sourceColName the input scalar column name, containing the category index to encode
    * @param categoriesNumber the number of categories overall, which will be the size of encoded Seq
    */
  def addCategoricalBinaryCoding(outColName: String, sourceColName: String, categoriesNumber: Int) = {
    addUDFColumn(outColName, sourceColName, categoricalBinaryEncodingUDF(categoriesNumber))
  }

  /**
    * Transforms a Seq[Seq[A]] column into a Seq[A] column, after flattening the inner Seq.
    * @param outColName the flattened output Seq[A] column name
    * @param sourceColName the nested Seq[Seq[A]] input column name
    */
  def addFlatten(outColName: String, sourceColName: String) = {
    TSS(series.withColumn(outColName, flatten(series(sourceColName))))
  }

  /**
    * Adds a Seq[Double] column containing the PCA coordinates obtained by applying the PCA algorithm on an input
    * Seq[Double] column. The methods also allows to store PCA byproducts into external files.
    * This variant relies on the Spark ML PCA implementation
    * @param outColName the added PCA coordinates column name
    * @param inColName the input column used for PCA computation
    * @param maxK the maximum number of principal components to compute (if significancy threshold is not reached before)
    * @param significancyThreshold the cumulative variance explanation ratio to reach for components number choice (if possible with less than <maxK> variables)
    * @param pcaLoadingsOutFile (optional) the File where to store the PCA Loadings
    * @param pcaVariancesOutFile (optional) the File where to store the PCA Variance per variable
    * @param pcaCos2OutFile (NOT IMPLEMENTED FOR NOW cause not supported by Spark ML) the File where to store the cos² matrix
    * @param pcaContribVariablesOutFile (NOT IMPLEMENTED FOR NOW cause not supported by Spark ML) the File where to store the per variable contribution to PCA axes matrix
    * @return a TSS with the extra PCA ooordinates column creation planned
    */
  def addPCA(outColName: String, inColName: String, maxK: Int, significancyThreshold: Double,
             pcaLoadingsOutFile: Option[File] = None, pcaVariancesOutFile: Option[File] = None,
             pcaCos2OutFile: Option[File] = None, pcaContribVariablesOutFile: Option[File] = None) = {
    val pca = new PCA()
      .setInputCol(inColName)
      .setOutputCol(outColName)
      .setK(maxK)
      .fit(series)
    val significantPCADimensions =
      pca.explainedVariance.values.indices.find(i => {
        pca.explainedVariance.values.slice(0, i + 1).sum >= significancyThreshold
      }).getOrElse(maxK - 1) + 1
    val pcaRes = new PCA()
      .setInputCol(inColName)
      .setOutputCol(outColName)
      .setK(significantPCADimensions)
      .fit(series)
    pcaLoadingsOutFile.map(f => {
      val parLoadingsFileWriter = new BufferedWriter(new FileWriter(f))
      (0 until pcaRes.pc.numRows).foreach(i => {
        val row = (0 until pcaRes.pc.numCols).map(j => {
          pcaRes.pc.apply(i, j)
        })
        parLoadingsFileWriter.write(i + "," + row.mkString(",") + "\n")
      })
      parLoadingsFileWriter.close()
    })
    pcaVariancesOutFile.map(f => {
      val parVarianceFileWriter = new BufferedWriter(new FileWriter(f))
      pcaRes.explainedVariance.values.indices.foreach(i => {
        parVarianceFileWriter.write(i + "," + pcaRes.explainedVariance.values(i) + "\n")
      })
      parVarianceFileWriter.close()
    })
    new TSS(pcaRes.transform(series))
  }

  /**
    * Adds a Seq[Double] column containing the PCA coordinates obtained by applying the PCA algorithm on an input
    * Seq[Double] column. The methods also allows to store PCA byproducts into external files.
    * This variant relies on the Scala SMILE PCA Algorithm, thus not cluster-ready. However, it allows to obtain
    * Cos² and variable contribution to axes unlike addPCA function.
    * @param outColName the added PCA coordinates column name
    * @param inColName the input column used for PCA computation
    * @param significancyThreshold the cumulative variance explanation ratio to reach for components number choice
    * @param pcaLoadingsOutFile (optional) the File where to store the PCA Loadings
    * @param pcaVariancesOutFile (optional) the File where to store the PCA Variance per variable
    * @param pcaCos2OutFile (optional) the File where to store the cos² matrix
    * @param pcaContribVariablesOutFile (optional) the File where to store the per variable contribution to PCA axes matrix
    * @return a TSS with the extra PCA ooordinates column creation planned
    */
  def addPCASmile(outColName: String, inColName: String, correlation: Boolean, significancyThreshold: Double,
                  pcaLoadingsOutFile: Option[File] = None, pcaVariancesOutFile: Option[File] = None,
                  pcaCos2OutFile: Option[File] = None, pcaContribVariablesOutFile: Option[File] = None,
                  pcaProjectionMatrixOutFile: Option[File] = None) = {
    assert(significancyThreshold >= 0D && significancyThreshold <= 1.0D)
    val seriesSub = series.orderBy(TSS.ID_COLNAME).select(TSS.ID_COLNAME, inColName)
    val collectedSub = seriesSub.map(x => (x.getSeq[Double](1).toArray, x.getLong(0))).collect()
    val inputCoordinatesForPCA = collectedSub.map(_._1)
    val tssIndices = collectedSub.map(_._2)
    val pcaRes = projection.pca(inputCoordinatesForPCA, correlation)
    val significantPCADimensions =
      pcaRes.getCumulativeVarianceProportion.indices.find(i => {
        pcaRes.getCumulativeVarianceProportion()(i) >= significancyThreshold
      }).get + 1
    pcaRes.setProjection(significantPCADimensions)
    val pcaProjection = pcaRes.project(inputCoordinatesForPCA)
    val parPCAProjection =
      series.sparkSession.sparkContext.parallelize(pcaProjection.zip(tssIndices)).toDF(outColName, "idjoin")
    val joinedTSS =
      addByLeftJoin(
        parPCAProjection,
        series(TSS.ID_COLNAME) === parPCAProjection("idjoin")
      ).drop("idjoin")
    pcaLoadingsOutFile.map(f => {
      val parLoadingsFileWriter = new BufferedWriter(new FileWriter(f))
      pcaRes.getLoadings.array().zipWithIndex.foreach(x => {
        parLoadingsFileWriter.write(x._2 + "," + x._1.mkString(",") + "\n")
      })
      parLoadingsFileWriter.close()
    })
    pcaVariancesOutFile.map(f => {
      val parVarianceFileWriter = new BufferedWriter(new FileWriter(f))
      pcaRes.getVariance.zipWithIndex.foreach(x => {
        parVarianceFileWriter.write(x._2 + "," + x._1 + "\n")
      })
      parVarianceFileWriter.close()
    })
    pcaCos2OutFile.map(f => {
      val pcaCos2FileWriter = new BufferedWriter(new FileWriter(f))
      val pcaCoordinateSqNorms = pcaProjection.map(x => {
        val n = breeze.linalg.norm(new DenseVector(x))
        n * n
      })
      pcaProjection.indices.foreach(i => {
        pcaCos2FileWriter.write(
          pcaProjection(i).indices.map(j => {
            scala.math.pow(pcaProjection(i)(j), 2) / pcaCoordinateSqNorms(i)
          }).mkString(",") + "\n"
        )
      })
    })
    pcaContribVariablesOutFile.map(f => {
      val pcaCoordinateVarWriter = new BufferedWriter(new FileWriter(f))
      val pcaCoordinateVarSqNorms = pcaProjection(0).indices.map(i => {
        val n = breeze.linalg.norm(new DenseVector(pcaProjection.map(x => x(i))))
        n * n
      })
      pcaProjection.indices.foreach(i => {
        pcaCoordinateVarWriter.write(
          pcaProjection(i).indices.map(j => {
            scala.math.pow(pcaProjection(i)(j), 2) / pcaCoordinateVarSqNorms(j)
          }).mkString(",") + "\n"
        )
      })
    })
    pcaProjectionMatrixOutFile.map(f => {
      val writer = new BufferedWriter(new FileWriter(f))
      pcaRes.getProjection.array().foreach(x => {
        writer.write(x.mkString(",") + "\n")
      })
      writer.close()
    })
    joinedTSS
  }

  /**
    * Adds a Seq[Double] column containing the MDS coordinates obtained by applying the MDS algorithm on an input
    * Seq[Double] column.
    * This variant relies on the Scala SMILE MDS Algorithm, thus not cluster-ready and potential bottleneck if huge data
    * @param outColName the added MDS coordinates column name
    * @param inColName the input column used for MDS computation
    * @param k the number of MDS coordinate dimensions
    * @param add whether to add a constant automatically estimated to make the learning matrix positive semi-definite
    * @param dist the distance to use between Seq[Double] descriptions of individuals to project
    * @param significancyThreshold the cumulative variance explanation ratio to reach for components number choice
    * @param eigenValuesOutFile (optional) the File where to store decomposition eigenvalues
    * @param varianceProportionOutFile (optional) the File where to store explained variance proportion of latent variables
    * @param ss the SparkSession
    * @return a TSS with the extra MDS column planned to be computed
    */
  def addMDSSmile[A](outColName: String, inColName: String, k: Int, add: Boolean, dist: (Seq[A], Seq[A]) => Double,
                     significancyThreshold: Double, eigenValuesOutFile: Option[File] = None, varianceProportionOutFile: Option[File] = None)
                    (implicit ss: SparkSession) = {
    val orderedIds = series.orderBy(TSS.ID_COLNAME).select(TSS.ID_COLNAME).map(_.getLong(0)).collect()
    val dists = rowDistances(inColName, dist).getAs2DArray
    val mdsRes = mds.mds(dists, scala.math.min(k, count - 1).toInt, add)
    val mdsVarProportions = mdsRes.getProportion
    val mdsCumVarProportions = mdsVarProportions.scanLeft(0D)(_ + _).drop(1)
    val significantDimensions =
      mdsCumVarProportions.indices.find(i => {
        mdsCumVarProportions(i) >= significancyThreshold
      }).get + 1
    val finalMdsRes = mds.mds(dists, significantDimensions, add)
    eigenValuesOutFile.map(f => {
      val writer = new BufferedWriter(new FileWriter(f))
      writer.write(finalMdsRes.getEigenValues.mkString(","))
      writer.close()
    })
    varianceProportionOutFile.map(f => {
      val writer = new BufferedWriter(new FileWriter(f))
      writer.write(finalMdsRes.getProportion.mkString(","))
      writer.close()
    })
    val outCoordinates = finalMdsRes.getCoordinates
    val outCoordinatesDF = ss.sparkContext.parallelize(orderedIds.zip(outCoordinates)).toDF("idJoin", outColName)
    addByLeftJoin(outCoordinatesDF, outCoordinatesDF("idJoin") === series(TSS.ID_COLNAME)).drop("idJoin")
  }

  /**
    * Adds a Seq[Double] column containing the MDS coordinates obtained by applying the MDS algorithm on an input
    * Seq[Double] column.
    * This variant is a distributed thus cluster-ready version, allowing to work on big matrices.
    * Note that since MDS requires to perform partial SVD on distances matrix, it can be quite costly for
    * an high enough number of individuals, and the choice should depend on the comparison between computational cluster size
    * and dataset size.
    * @param outColName the added MDS coordinates column name
    * @param inColName the input column used for MDS computation
    * @param k the number of maximal MDS coordinate dimensions
    * @param dist the distance to use between Seq[Double] descriptions of individuals to project
    * @param significancyThreshold the cumulative variance explanation ratio to reach for components number choice
    * @param innerPartitionsNumber the number of partitions to use during the SVD computation
    * @param runsNb the number of runs to execute before choosing the best MDS transformation (the one with lowest stress function)
    * @param noiseFactor the maximal factor to multiply to the minimal non zero value in the matrix before SVD computation then add to the matrix diagonal
    *                    in order to force optimization to provide various results whenever requesting <runsNb> > 1
    * @param eigenValuesOutFile (optional) the File where to store decomposition eigenvalues
    * @param varianceProportionOutFile (optional) the File where to store explained variance proportion of latent variables
    * @param stressOutFile (optional) the File where to store final stress double
    * @param ss the SparkSession
    * @return a TSS with the extra MDS column planned to be computed
    */
  def addMDS[A](outColName: String, inColName: String, k: Int, dist: (Seq[A], Seq[A]) => Double,
                significancyThreshold: Double, innerPartitionsNumber: Int, runsNb: Int = 1, noiseFactor: Double = 0D,
                eigenValuesOutFile: Option[File] = None, varianceProportionOutFile: Option[File] = None, stressOutFile: Option[File] = None)
               (implicit ss: SparkSession) = {

    //Only work on requested column, keep ids for sparse distance implementation
    val subSeries = series.select(TSS.ID_COLNAME, inColName).repartition(innerPartitionsNumber)//.persist()
    subSeries.createOrReplaceTempView("subSeries")

    /*println("SS")
    subSeries.show(100)*/

    val crossSeries =
      ss.sql("SELECT * FROM subSeries ss1, subSeries ss2 WHERE ss1.id <= ss2.id")
        .repartition(innerPartitionsNumber/*, col("ss1.id")*/)

    /*println("CS")
    crossSeries.show(100)*/

    //Compute pairwise distances
    val ds = crossSeries.map(x => {
      val d = dist(x.getSeq[A](1), x.getSeq[A](3))
      IndexedDistance(x.getLong(0), x.getLong(2), -0.5 * d * d)
    }).persist

    /*println("DS")
    ds.show(100)*/

    //Build the symmetric distance matrix then perform row means and overall mean computations
    val fullDSTemp =
      ds.union(ds.select(ds("id2").alias("id1"), ds("id1").alias("id2"), ds("distance")).as[IndexedDistance].where(col("id1") =!= col("id2")))
    val rowMeans = fullDSTemp.groupBy("id1").agg(mean("distance").alias("mean")).coalesce(innerPartitionsNumber).persist
    val fullMean = rowMeans.agg(mean("mean")).first.getDouble(0)

    /*println("FM")
    println(fullMean)

    println("RM")
    rowMeans.toDF.orderBy("id1").show(100)

    println("FDS")
    fullDSTemp.orderBy("id1", "id2").show(10000)*/

    //Normalize distances by global mean and per row means
    ds.toDF.select(col("id1"), col("id2"), (col("distance") + lit(fullMean)).alias("distance")).createOrReplaceTempView("ds")
    rowMeans.createOrReplaceTempView("rowMeans")
    val odists =
      ss.sql("SELECT ds.id1 AS id1, ds.id2 AS id2, ds.distance - rm1.mean - rm2.mean AS distance FROM ds, rowMeans AS rm1, rowMeans AS rm2 " +
        "WHERE ds.id1 == rm1.id1 AND ds.id2 == rm2.id1").as[IndexedDistance].coalesce(innerPartitionsNumber).persist
    val minValue = odists.select("distance").where(col("distance") > 0).agg(min("distance")).first.getDouble(0)
    val maxOffsetBound = minValue * noiseFactor
    println("maxOffsetBound: " + maxOffsetBound)

    /*println("OD")
    odists.toDF.show(100)*/

    //Build the final symmetric normalized distances
    val symDS = odists.select(odists("id2").alias("id1"), odists("id1").alias("id2"), odists("distance")).as[IndexedDistance]
    val fullDS = odists.union(symDS.where(symDS("id1") =!= symDS("id2"))).orderBy(col("id1"), col("id2")).coalesce(innerPartitionsNumber).persist

    //Compute the overall matrix Frobenius norm for later variable contributions computations
    val sqrFrobNorm = fullDS.agg(sum(pow(col("distance"), 2))).first.getDouble(0)

    import ss.implicits._

    //Build a distributed dense matrix from symmetric distances computations
    /*println("TSDS")
    TSDS(fullDS, count.toInt).orderByIds.indexedDistances.show(1000)

    println("GD")
    TSDS(fullDS, count.toInt).getAsGroupedLists(TSDS.ID1_COLNAME).orderBy("refId").rdd.take(10).foreach(gd => {
      println(gd.refId)
      println(gd.distances.mkString(","))
    })*/

    def computeSVD = {
      val randFactor = new scala.util.Random().nextDouble()
      val offset = randFactor * maxOffsetBound
      val diagonalFullDS = fullDS.where(fullDS("id1") === fullDS("id2")).map(x => x.copy(distance = x.distance + offset))
      val nonDiagonalFullDS = fullDS.where(fullDS("id1") !== fullDS("id2"))

      //This step seems to explain variability in results ...
      val groupedDistsDS = TSDS(diagonalFullDS.union(nonDiagonalFullDS), count.toInt).getAsGroupedLists(TSDS.ID1_COLNAME).orderBy("refId")

      val groupedDists = groupedDistsDS.rdd//.persist
      //Force columns to be sorted according to id2
      val dists = groupedDists.map(x => {
        org.apache.spark.mllib.linalg.Vectors.dense(x.distances.sortWith(_._1 < _._1).map(_._2).toArray)
      }).persist

      val rowMat = new RowMatrix(dists, count, count.toInt)

      //Compute eigenvector decompositions = SVD decomposition (with eigenvalues being square root of singular values)
      val svd = rowMat.computeSVD(k, false)

      //Compute squares of singular values to compare them with Frobenius norm of the distances matrix
      //And thus obtain variance contributions of each singular / eigenvector
      val svdSigmaSquared = svd.s.toArray.map(x => (x - offset) * (x - offset))
      val mdsVarProportions = svdSigmaSquared.map(_ / sqrFrobNorm)
      val mdsCumVarProportions = mdsVarProportions.scanLeft(0D)(_ + _).drop(1)

      //Get Significant Number of Dimensions (so that exceeding the threshold)
      val significantDimensions =
        mdsCumVarProportions.indices.find(i => {
          mdsCumVarProportions(i) >= significancyThreshold
        }).map(_ + 1).getOrElse(k)

      //Select subset of eigenvectors depending on desired significancy threshold
      val finalEigenvalues = svd.s.toArray.take(significantDimensions).map(x => scala.math.sqrt(x - offset))

      //Get the final points coordinates in reduced space by a truncated matrix multiplication of significant eigenvectors and eigenvalues
      val outCoordinates = svd.V.rowIter.map(r => r.toArray.take(significantDimensions).zip(finalEigenvalues).map{
        case (x, y) => x * y
      })

      /*println("OutCoord")
      println(svd.V.rowIter.map(r => r.toArray.take(significantDimensions).zip(finalEigenvalues).map{
        case (x, y) => x * y
      }).map(_.mkString(",")).mkString("\n"))*/

      val outCoordinatesDF = ss.sparkContext.parallelize(groupedDists.map(_.refId).collect.zip(
        outCoordinates.toSeq
      )).toDF("idJoin", outColName)

      //Compute stress (sum of squared distances between full and reduced representations)
      val outCoordinatesDistances =
      TSS.distances(outCoordinatesDF.as[(Long, Seq[Double])], TSS.euclideanF[Seq[Double]](false))
        .indexedDistances.withColumnRenamed("distance", "distance_")
      val outStressDF = ds.join(outCoordinatesDistances,
        outCoordinatesDistances(TSDS.ID1_COLNAME) === ds("id1") && outCoordinatesDistances(TSDS.ID2_COLNAME) === ds("id2"), "left_outer")
        .select(functions.pow(col("distance_") - functions.sqrt(col("distance") / lit(-0.5)), lit(2D)).alias("d"))
      val outStress = outStressDF.agg(sum("d")).first.getDouble(0)

      dists.unpersist(false)

      (outStress, finalEigenvalues, mdsVarProportions, significantDimensions, outCoordinatesDF)
    }

    //Execute all runs and keep results of the one with lowest stress
    //Use vars to avoid storing all SVD simultaneously
    val (outStress, finalEigenvalues, mdsVarProportions, significantDimensions, outCoordinatesDF) = computeSVD
    var (bestStress, bestFinalEigenvalues, bestMDSVarProportions, bestSignificantDimensions, bestOutCoordinatesDF) =
      (outStress, finalEigenvalues, mdsVarProportions, significantDimensions, outCoordinatesDF)
    println("Iteration Out Stress: " + outStress)
    var remainingRuns = runsNb - 1
    while(remainingRuns > 0){
      val (outStress, finalEigenvalues, mdsVarProportions, significantDimensions, outCoordinatesDF) = computeSVD
      println("Iteration Out Stress: " + outStress)
      if(outStress < bestStress) {
        bestStress = outStress
        bestFinalEigenvalues = finalEigenvalues
        bestSignificantDimensions = significantDimensions
        bestOutCoordinatesDF = outCoordinatesDF
        bestMDSVarProportions = mdsVarProportions
      }
      remainingRuns -= 1
    }

    /*println("SVD::S")
    println(svd.s.toArray.mkString(","))
    println("SVD::V")
    println(svd.V.rowIter.map(_.toArray.mkString(",")).mkString("\n"))*/

    //Save model:
    // - significant dimension eigenvalues (square root of corresponding singular values)
    // - corresponding features contributions to overall variance
    eigenValuesOutFile.map(f => {
      val writer = new BufferedWriter(new FileWriter(f))
      writer.write(bestFinalEigenvalues.mkString(","))
      writer.close()
    })
    varianceProportionOutFile.map(f => {
      val writer = new BufferedWriter(new FileWriter(f))
      writer.write(bestMDSVarProportions.take(bestSignificantDimensions).mkString(","))
      writer.close()
    })

    //TODO: Output to File
    println("OUT STRESS = " + bestStress)
    stressOutFile.map(f => {
      val writer = new BufferedWriter(new FileWriter(f))
      writer.write(bestStress.toString)
      writer.close()
    })

    //Clean Cache
    ds.unpersist
    rowMeans.unpersist
    odists.unpersist
    //dddists.unpersist(false)
    //groupedDists.unpersist(false)
    fullDS.unpersist(false)

    //Join to TSS object to allow chaining
    addByLeftJoin(bestOutCoordinatesDF, bestOutCoordinatesDF("idJoin") === series(TSS.ID_COLNAME)).drop("idJoin")
  }

  /**
    * Adds new columns by left join with an external DataFrame.
    * @param sourceDataFrame the external DataFrame to join with "this" one
    * @param joinExpr the join expression between inner "this" DataFrame and external one
    * @return a TSS with extra columns of external DataFrame planned to be processed, given the join condition.
    *         The output is a left outer join, so that all original "this" row still exist afterwards.
    */
  //Assumes join is performed by id and out column names are taken from sourceDataFrame
  def addByLeftJoin(sourceDataFrame: DataFrame, joinExpr: Column) = {
    // = series(com.github.acout.spark.tss.core.TSS.ID_COLNAME) === sourceDataFrame(com.github.acout.spark.tss.core.TSS.ID_COLNAME)
    new TSS(series.join(sourceDataFrame, joinExpr, "left_outer"))
  }

  /**
    * Compute PCA correlation circle matrix and return it as a DataFrame with one column per
    * (Feature, PCA coordinate) pair. The correlation coefficients are computed by subsets to avoid
    * Spark issues due to too big queries. The batch size is defined by an optional param.
    * @param pcaColName the Seq[Double] column name containing the pca coordinates of each row
    * @param originalFeaturesColName the Seq[Double] column name containing the original features of each row
    * @param corrPerSelect (optional) the number of correlation coefficients to compute per select (<= features number * pca coordinates number).
    *                      The bigger this value, the more likely Spark will throw an Exception about too big query size. The smaller the value,
    *                      the longer the computation will be (due to more join operations and less parallelism).
    * @return a PCA correlation circle matrix as a DataFrame with one column per (Feature, PCA coordinate) pair,
    *         each named "corr.f.<feature_id>.r.<pcaCoordinate_id>"
    */
  def pcaCorrelationCircle(pcaColName: String, originalFeaturesColName: String, corrPerSelect: Int = 1000) = {
    val pcaSize = series.select(pcaColName).first().getSeq[Double](0).size
    val featuresSize = series.select(originalFeaturesColName).first().getSeq[Double](0).size
    val pcaDF = series.select((0 until pcaSize).map(i => {
      element_at(series(pcaColName), i + 1).alias(pcaColName + i)
    }) :+ series(TSS.ID_COLNAME).alias("idPCA"): _*)
    val featuresDF = series.select((0 until featuresSize).map(i => {
      element_at(series(originalFeaturesColName), i + 1).alias(originalFeaturesColName + i)
    }) :+ series(TSS.ID_COLNAME).alias("idFeats"): _*)
    pcaDF.persist()
    pcaDF.take(1) //Guarantee persist is called before join
    val joinedDF = pcaDF.join(featuresDF, pcaDF("idPCA") === featuresDF("idFeats")).drop("idFeats")
    //joinedDF.groupBy("idPCA").pivot("idPCA").agg((pcaColName + "1", originalFeaturesColName + "1")).show(10)
    //try{
    val corrPairs = for (f <- 0 until featuresSize; p <- 0 until pcaSize) yield (f, p)
    val corrCols = (0 until pcaSize).flatMap(p => {
      (0 until featuresSize).map(f => {
        ("corr.f" + f + ".r" + p, corr(joinedDF(originalFeaturesColName + f), joinedDF(pcaColName + p)).alias("corr.f" + f + ".r" + p))
      })
    })
    val intermDFs = (0 until corrCols.size by corrPerSelect).map(i => {
      joinedDF.select(corrCols.slice(i * corrPerSelect, scala.math.min((i + 1) * corrPerSelect, corrCols.size)).map(_._2): _*)
    })
    val res = intermDFs.foldLeft(series.sparkSession.emptyDataFrame){(df, ddf) =>
      df.crossJoin(ddf)
    }
    res.take(1)
    pcaDF.unpersist()
    res
    /*}catch{
      case _: Throwable => {
        val corrPairs = for (f <- 0 until featuresSize; p <- 0 until pcaSize) yield (f, p)
        val res = corrPairs.foldLeft(joinedDF) { (df, c) =>
          df.withColumn("corr.f" + c._1 + ".r" + c._2, corr(col(originalFeaturesColName + c._1), col(pcaColName + c._2)))
        }
        res
      }
    }*/
  }

  /**
    * Compute contingency matrix between a Map[String, String] and a scalar column as a DataFrame.
    * @param out1KeyColName the output key column name for input Map[String, String] keys
    * @param out1ValColName the output value column name for input Map[String, String] values
    * @param out2ColName the output value column name for input scalar values
    * @param outFreqColName the output column name storing the actual frequencies for a given tuple of labels
    * @param mapColName the input Map[String, String] column name
    * @param scalarColName the input scalar column name
    * @param ss the SparkSession object
    * @return a DataFrame with 4 columns, the 3 first corresponding to the contingency context (pair of key, value for
    *         Map, and an extra value for the scalar) and the last corresponding to the actual associated frequency.
    */
  def contingencyMatrixMapScalar(out1KeyColName: String, out1ValColName: String, out2ColName: String, outFreqColName: String, mapColName: String, scalarColName: String)(implicit ss: SparkSession) = {
    def mapScalarTuples[A <: AnyVal](m: Map[String, String], s: A) =  {
      m.keys.map(k => (k, m(k), s, 1))
    }
    series.select(mapColName, scalarColName).flatMap(x => {
      val m = x.getMap[String, String](0)
      mapScalarTuples(m.toMap, x.getInt(1))
    }).toDF(out1KeyColName, out1ValColName, out2ColName, outFreqColName)
      .groupBy(out1KeyColName, out1ValColName, out2ColName).agg(sum(outFreqColName).alias(outFreqColName))
  }
  /**
    * Compute contingency matrix between two scalar columns.
    * @param out1ColName the output value column name for first input scalar values
    * @param out2ColName the output value column name for second input scalar values
    * @param outFreqColName the output column name storing the actual frequencies for a given tuple of labels
    * @param scalarColName1 the first input scalar column name
    * @param scalarColName2 the second input scalar column name
    * @param ss the SparkSession object
    * @return a DataFrame with 3 columns, the 2 first corresponding to the contingency context (pair of scalar values)
    *         and the last corresponding to the actual associated frequency.
    */
  def contingencyMatrixScalarScalar(out1ColName: String, out2ColName: String, outFreqColName: String, scalarColName1: String, scalarColName2: String)(implicit ss: SparkSession) = {
    def scalarScalarTuples[A <: AnyVal, B <: AnyVal](s1: A, s2: B) =  {
      (s1, s2, 1)
    }
    series.select(scalarColName1, scalarColName2).map(x => {
      scalarScalarTuples(x.getInt(0), x.getInt(1))
    }).toDF(out1ColName, out2ColName, outFreqColName)
      .groupBy(out1ColName, out2ColName).agg(sum(outFreqColName).alias(outFreqColName))
  }

  /**
    * Compute contingency matrix between 2 Map[String, String]
    * @param out1KeyColName the output key column name for the first input Map[String, String] keys
    * @param out1ValColName the output value column name for the first input Map[String, String] values
    * @param out2KeyColName the output key column name for the second input Map[String, String] keys
    * @param out2ValColName the output value column name for the second input Map[String, String] values
    * @param outFreqColName the output column name storing the actual frequencies for a given tuple of labels
    * @param mapColName1 the first input Map[String, String] column name
    * @param mapColName2 the second input Map[String, String] column name
    * @param ss the SparkSession object
    * @return @return a DataFrame with 5 columns, the 4 first corresponding to the contingency context (pair of key, value for
    *         each Map) and the last corresponding to the actual associated frequency.
    */
  def contingencyMatrixMapMap(out1KeyColName: String, out1ValColName: String, out2KeyColName: String, out2ValColName: String,
                              outFreqColName: String, mapColName1: String, mapColName2: String)(implicit ss: SparkSession) = {
    import ss.implicits._
    //TODO: propose more generic methods
    def mapMapTuples(m1: Map[String, String], m2: Map[String, String]) =  {
      m1.keys.flatMap(k1 => {
        m2.keys.map(k2 => {
          (k1, m1(k1), k2, m2(k2), 1)
        })
      })
    }
    series.select(mapColName1, mapColName2).flatMap(x => {
      mapMapTuples(x.getMap[String, String](0).toMap, x.getMap[String, String](1).toMap)
    }).toDF(out1KeyColName, out1ValColName, out2KeyColName, out2ValColName, outFreqColName)
      .groupBy(out1KeyColName, out1ValColName, out2KeyColName, out2ValColName).agg(sum(outFreqColName).alias(outFreqColName))
  }

  /**
    * Adds a column with per-row clustering membership after a K-Means algorithm execution.
    * This variant uses Scala SMILE K-Means implementation, thus is not cluster ready.
    * @param outColName the added column name
    * @param sourceColName the input Seq[Double] column containing features to use for clustering
    * @param k the number of clusters
    * @param epsilon the convergence threshold
    * @param iterMax the max number of iterations
    * @param runs the number of (random) restarts, to avoid local minima better
    * @param centersOutFile (optional) the File where to store prototype final coordinates
    * @param ss the SparkSession
    * @return a TSS with the added column planned to process
    */
  def addKMeansSmile(outColName: String, sourceColName: String, k: Int, epsilon: Double, iterMax: Int, runs: Int = 1, centersOutFile: Option[File] = None)(implicit ss: SparkSession) = {
    val seriesSub = series.orderBy(TSS.ID_COLNAME).select(TSS.ID_COLNAME, sourceColName)
    val collectedSub = seriesSub.map(x => (x.getSeq[Double](1).toArray, x.getLong(0))).collect()
    val inputCoordinates = collectedSub.map(_._1)
    val tssIndices = collectedSub.map(_._2)
    val res = clustering.kmeans(inputCoordinates, k, iterMax, runs)
    val centers = res.centroids
    centersOutFile.map(f => {
      val writer = new BufferedWriter(new FileWriter(f))
      centers.indices.foreach(i => {
        writer.write(i + "," + centers(i).mkString(",") + "\n")
      })
      writer.close()
    })
    val clusterIds = res.getClusterLabel()
    val parClusterIds =
      series.sparkSession.sparkContext.parallelize(clusterIds.zip(tssIndices)).toDF(outColName, "idjoin")
    val joinedTSS =
      addByLeftJoin(
        parClusterIds,
        series(TSS.ID_COLNAME) === parClusterIds("idjoin")
      )
    joinedTSS.drop("idjoin")
  }

  /**
    * Adds a column with per-row clustering membership after an Ascending Hierarchical Clustering (AHC) algorithm execution.
    * This variant uses Scala SMILE implementation, thus is not cluster ready. To capitalize on computations, this method
    * allows to cut the learned tree at different k values to generate as many output columns.
    * @param outColNameFun the function binding a k value to a column name
    * @param sourceColName the input column containing individual coordinates to use for clustering
    * @param ks the different cluster numbers k to evaluate on the learned AHC tree
    * @param linkage the linkage method as a String understood by SMILE hclust function
    * @param dist the distance function to generate the distance matrix used as AHC input
    * @param treeOutFile (optional) the output File where the tree model itself will be stored
    * @tparam A the vector type of individual descriptions used for clustering
    * @return a TSS with the added column planned to process
    */
  def addHClustSmile[A](outColNameFun: Int => String, sourceColName: String, ks: Array[Int], linkage: String,
                        dist: (Seq[A], Seq[A]) => Double, treeOutFile: Option[File] = None) = {
    val tssIndices = series.orderBy(TSS.ID_COLNAME).select(TSS.ID_COLNAME).map(_.getLong(0)).collect()
    val dists = rowDistances(sourceColName, dist).getAs2DArray
    val res = clustering.hclust(dists, linkage)
    treeOutFile.map(f => {
      val writer = new BufferedWriter(new FileWriter(f))
      res.getTree.foreach(t => {
        writer.write(t.mkString(",") + "\n")
      })
      writer.close()
    })
    val membershipsByK = ks.map(res.partition(_))
    val membershipsById = tssIndices.indices.map(i => membershipsByK.indices.map(ki => ks(ki) -> membershipsByK(ki)(i)).toMap)
    val parClusterIds = series.sparkSession.sparkContext.parallelize(membershipsById.zip(tssIndices))
      .toDF(outColNameFun(0), "idjoin")
      .select(ks.map(k => element_at(col(outColNameFun(0)), k).alias(outColNameFun(k))) ++ Array(col("idjoin")): _*)
    val joinedTSS = addByLeftJoin(
      parClusterIds,
      series(TSS.ID_COLNAME) === parClusterIds("idjoin")
    )
    joinedTSS.drop("idjoin")
  }

  /**
    * Retrieves a mapping between cluster labels of a given clustering, and associated vector barycenter of the individuals in this cluster
    * @param coordinatesColName the column name containing the vector coordinates of individuals on which performing per index means
    * @param clusteringMembershipColName the column name containing the clustering membership for each individual (a scalar)
    * @return a DataFrame with 2 columns: "cluster" contains the cluster index, and "barycenter" contains the cluster barycenter as a mean vector of coordinates
    */
  def getClusteringBarycenters(coordinatesColName: String, clusteringMembershipColName: String): DataFrame = {
    val keyRDD = series.select(coordinatesColName, clusteringMembershipColName).rdd.map(x => (x.getInt(1), x.getSeq[Double](0)))
    type C = (Seq[Double], Int)
    def init(s: Seq[Double]) = (s, 1)
    def reduceItemBuffer(x: C, s: Seq[Double]) = (x._1.indices.map(i => x._1(i) + s(i)), x._2 + 1)
    def reduceBufferBuffer(c1: C, c2: C) = (c1._1.indices.map(i => c1._1(i) + c2._1(i)), c1._2 + c2._2)
    keyRDD.combineByKey(init _, reduceItemBuffer _, reduceBufferBuffer _).mapValues{
      case (sums, length) => {
        sums.indices.map(i => sums(i) / length)
      }
    }.toDF("cluster", "barycenter")
  }

  /**
    * Returns the Davies-Bouldin internal clustering evaluation measure
    * @param representationColName the column name containing the vector coordinates of individuals for distance computations
    * @param clusteringColName the column name containing the clustering id of individuals for clustering evaluation
    * @param ss
    * @return the Davies-Bouldin score for given clustering, using given points representation
    */
  def daviesBouldin(representationColName: String, clusteringColName: String)(implicit ss: SparkSession): Double = {
    val clusterizedRaw = series.select(TSS.ID_COLNAME, representationColName, clusteringColName).rdd.map(x => {
      EasyClusterizable(x.getLong(0),
        ScalarVector(x.getSeq[Double](1).toArray)
      ).addClusterIDs(x.getInt(2))
    })
    val metric = new Euclidean(false)
    return InternalIndicesDistributed.daviesBouldin(ss.sparkContext, clusterizedRaw, metric, 0)
  }

  /**
    * Returns the Ball Hall internal clustering evaluation measure
    * @param representationColName the column name containing the vector coordinates of individuals for distance computations
    * @param clusteringColName the column name containing the clustering id of individuals for clustering evaluation
    * @param ss
    * @return the Ball Hall score for given clustering, using given points representation
    */
  def ballHall(representationColName: String, clusteringColName: String)(implicit ss: SparkSession): Double = {
    val clusterizedRaw = series.select(TSS.ID_COLNAME, representationColName, clusteringColName).rdd.map(x => {
      EasyClusterizable(x.getLong(0),
        ScalarVector(x.getSeq[Double](1).toArray)
      ).addClusterIDs(x.getInt(2))
    })
    val metric = new Euclidean(false)
    return InternalIndicesDistributed.ballHall(clusterizedRaw, metric, 0)
  }

  /**
    * Adds two columns with hard and soft per-row clustering memberships after a Gaussian Mixture Model (GMM) algorithm execution.
    * This variant uses Spark ML GMM implementation.
    * @param predOutColName the added hard cluster membership scalar column name
    * @param probaOutColName the added soft cluster membership Vector column name
    * @param logLikelihoodColName the added per sample log likelihood Vector column name
    * @param sourceColName the input Spark ML Vector column containing features to use for clustering
    * @param k the number of clusters
    * @param maxIter the max number of iterations
    * @param initWithKMeans (optional) whether to use KMeans++ as a first step to initialize GMM (true by default; otherwise, random initialization is performed)
    * @param runsNb (optional) number of runs to compute before keeping the best model in terms of maximum log likelihood
    * @param centersOutFile (optional) the File where to store prototype final coordinates
    * @param fullModelOutFilePath (optional) the path where to store the full GMM Model in Spark serialization format
    * @return a TSS with the added columns planned to process
    */
  def addGMM(predOutColName: String, probaOutColName: String, logLikelihoodColName: String, sourceColName: String, k: Int, maxIter: Int, initWithKMeans: Boolean = true, runsNb: Int = 1,
             centersOutFile: Option[File] = None, fullModelOutFilePath: Option[String] = None) = {

    if(initWithKMeans){

      //Encapsulation inside a function to allow several runs to be executed
      def doIt = {

        val km = new KMeans().setK(k).setFeaturesCol(sourceColName)
          .setPredictionCol(predOutColName).setMaxIter(maxIter)
        val initKMModel = km.fit(series)
        val kmState = initKMModel.transform(series)

        import series.sqlContext.implicits._
        val kmClusterCentersDF = initKMModel.clusterCenters.zipWithIndex.toSeq.toDF("clusterCenter", "clusterId").persist
        kmClusterCentersDF.take(1) //Trigger persist

        //Adapt k value using the one returned by KMeans init which can be lowered than desired number of clusters
        val realK = initKMModel.clusterCenters.size

        val joinedKMState = kmState.toDF.join(kmClusterCentersDF, kmState(predOutColName) === kmClusterCentersDF("clusterId"))

        //Select point - center for each point, then multiply result by its transpose
        val kmStateAgregates = joinedKMState.select("clusterCenter", sourceColName, predOutColName).rdd.map{
          case Row(cc: org.apache.spark.ml.linalg.Vector, x: org.apache.spark.ml.linalg.Vector, cid: Int) => {
            val normalizedX = x.toArray.zip(cc.toArray).map(a => a._1 - a._2)
            val mnx = new DenseMatrix(x.size, 1, normalizedX)
            val tmnx = mnx.t
            val covMat = mnx * tmnx
            (cid, (covMat, 1))
          }
          //Group By Cluster and Sum matrices ...
        }.reduceByKey{
          case ((a, s), (b, t)) => {
            (a + b, s + t)
          }
        }.sortByKey(true).persist

        val gaussianInitValues = kmStateAgregates.map{g =>
          (g._2._1 * (1.0 / g._2._2.toDouble), g._2._2.toDouble)
        }.collect.map(g => (g._1, g._2 / count.toDouble))
        kmClusterCentersDF.unpersist
        kmStateAgregates.unpersist(false)
        val gaussianInitWeights = gaussianInitValues.map(_._2)
        val gaussianInitMeans = initKMModel.clusterCenters
        //... then average values
        val gaussianInitCovMatrices = gaussianInitValues.map(_._1)
        val initGaussians = gaussianInitWeights.indices.map(i => {
          new org.apache.spark.mllib.stat.distribution.MultivariateGaussian(
            org.apache.spark.mllib.linalg.DenseVector.fromML(gaussianInitMeans(i).toDense), new org.apache.spark.mllib.linalg.DenseMatrix(
              gaussianInitCovMatrices(i).rows, gaussianInitCovMatrices(i).cols, gaussianInitCovMatrices(i).toArray
            )
          )
        }).toArray

        //build MLLib API GaussianMixtureModel with centers and computed covariance matrices
        val gmmInitModel = new org.apache.spark.mllib.clustering.GaussianMixtureModel(gaussianInitWeights, initGaussians)
        //build the GaussianMixture object from it (MLLib API ... Otherwise, not possible to initialize with custom model ...)
        val seriesRDD = series.select(sourceColName).rdd.map{
          case Row(v: org.apache.spark.ml.linalg.Vector) => org.apache.spark.mllib.linalg.DenseVector.fromML(v.toDense).asInstanceOf[org.apache.spark.mllib.linalg.Vector]
        }

        println("K: " + k + ", RK: " + realK)
        val mlLibModel = new org.apache.spark.mllib.clustering.GaussianMixture().setK(realK).setInitialModel(gmmInitModel).setMaxIterations(maxIter).run(seriesRDD)

        val resultDF =
          series.withColumn(predOutColName, TSS.predictUDFMLLib(mlLibModel)(series(sourceColName)))
            .withColumn(probaOutColName, TSS.softPredictUDFMLLib(mlLibModel)(series(sourceColName)))
            .withColumn(logLikelihoodColName, TSS.logLikelihoodIndividualMLLib(mlLibModel)(series(sourceColName)))

        (mlLibModel, resultDF.agg(sum(logLikelihoodColName)).map(_.getDouble(0)).first, resultDF)
      }//End doIt

      //Compute several runs and keep maximal loglikelihood one
      //(Use vars to avoid storing first all the runs then choosing the best one)
      val (mlLibModel, llSum, resultDF) = doIt
      var bestModel = mlLibModel
      var bestLLSum = llSum
      var bestResultDF = resultDF
      var remainingRunsNb = runsNb - 1
      println("Iteration Log Likelihood: " + llSum)
      while(remainingRunsNb > 0){
        val (mlLibModel, llSum, resultDF) = doIt
        println("Iteration Log Likelihood: " + llSum)
        if(llSum > bestLLSum){
          bestLLSum = llSum
          bestModel = mlLibModel
          bestResultDF = resultDF
        }
        remainingRunsNb -= 1
      }

      //Save byproducts (model information)
      centersOutFile.map(f => {
        val writer = new BufferedWriter(new FileWriter(f))
        bestModel.gaussians.zipWithIndex.foreach{case (g, i) => {
          writer.write(i + "," + g.mu.toArray.mkString(",") + "\n")
        }}
        writer.close()
      })
      fullModelOutFilePath.map(path => {
        val fs = FileSystem.get(series.sparkSession.sparkContext.hadoopConfiguration)
        val ppath = new Path(path)
        if (fs.exists(ppath))
          fs.delete(ppath, true)
        bestModel.save(series.sparkSession.sparkContext, path)
      })

      TSS(bestResultDF)
      //Convert to ML API to have unified prediction code in the end (and avoid extra join and log likelihood duplicate code ...)
      //NOT POSSIBLE (PRIVATE CTOR ...)
      /*new GaussianMixtureModel(UUID.randomUUID().toString, gaussianInitWeights, mlLibModel.gaussians.indices.map(i => {
        new org.apache.spark.ml.stat.distribution.MultivariateGaussian(
          mlLibModel.gaussians(i).mu.asML, mlLibModel.gaussians(i).sigma.asML
        )
      })).setFeaturesCol(sourceColName).setPredictionCol(predOutColName).setProbabilityCol(probaOutColName)*/
    }else{

      //Encapsulation inside a function to allow several runs to be executed
      def doIt = {
        val model = new GaussianMixture().setK(k).setFeaturesCol(sourceColName)
          .setPredictionCol(predOutColName).setProbabilityCol(probaOutColName)
          .setMaxIter(maxIter).fit(series)
        val transformedSeries = model.transform(series)
        val resultDF = transformedSeries.withColumn(logLikelihoodColName, TSS.logLikelihoodIndividualML(model)(transformedSeries(sourceColName)))
        (model, resultDF.agg(sum(logLikelihoodColName)).map(_.getDouble(0)).first, resultDF)
      }

      //Compute several runs and keep maximal loglikelihood one
      //(Use vars to avoid storing first all the runs then choosing the best one)
      val (mlLibModel, llSum, resultDF) = doIt
      var bestModel = mlLibModel
      var bestLLSum = llSum
      var bestResultDF = resultDF
      var remainingRunsNb = runsNb - 1
      println("Iteration Log Likelihood: " + llSum)
      while(remainingRunsNb > 0){
        val (mlLibModel, llSum, resultDF) = doIt
        println("Iteration Log Likelihood: " + llSum)
        if(llSum > bestLLSum){
          bestLLSum = llSum
          bestModel = mlLibModel
          bestResultDF = resultDF
        }
        remainingRunsNb -= 1
      }

      //Save byproducts (model information)
      centersOutFile.map(f => {
        val writer = new BufferedWriter(new FileWriter(f))
        bestModel.gaussians.zipWithIndex.foreach{case (g, i) => {
          writer.write(i + "," + g.mean.toArray.mkString(",") + "\n")
        }}
        writer.close()
      })
      fullModelOutFilePath.map(path => {
        bestModel.write.overwrite().save(path)
      })
      TSS(bestResultDF)
    }
  }

  def addAutoGMM(predOutColName: String, probaOutColName: String, logLikelihoodColName: String, sourceColName: String, kBounds: (Int ,Int), autoInitIter: Int, autoIter: Int, maxGMMIter: Int, initWithKMeans: Boolean = true, runsNb: Int = 1, centersOutFile: Option[File] = None, fullModelOutPath: Option[String] = None)(implicit ss: SparkSession) = {
    def target(vector: DenseVector[Double]) = {
      val k = vector(0)
      val randId = UUID.randomUUID().toString
      val tempTSS =
        addGMM(predOutColName + randId, probaOutColName + randId, logLikelihoodColName + randId, sourceColName, k.round.toInt, maxGMMIter, initWithKMeans, runsNb, None)
        .addSeqFromMLVector(sourceColName + randId, sourceColName)
      val db = tempTSS.daviesBouldin(sourceColName + randId, predOutColName + randId)
      -db
    }
    ???
    // val tuner = new BayesianTuner(target, Array((kBounds._1, kBounds._2, "discrete")), autoIter, autoInitIter)
    // val optimResult = tuner.tune()
    // val bestK = optimResult.x(0)
    // addGMM(predOutColName, probaOutColName, logLikelihoodColName, sourceColName, bestK.round.toInt, maxGMMIter, initWithKMeans, runsNb, centersOutFile, fullModelOutPath)
    ???
  }

  /**
    * Adds one column with hard membership after a SOM algorithm execution.
    * This variant uses Spark ML SOM implementation made by Florent Forest, soon maintained by the C4E project.
    * @param predOutColName the added hard cluster membership scalar column name
    * @param sourceColName the input Spark ML Vector column containing features to use for clustering
    * @param height the height of SOM model map (the number of cluster is obtained by height x width)
    * @param width the width of SOM model map (the number of cluster is obtained by height x width)
    * @param maxIter the max number of iterations
    * @param centersOutFile (optional) the File where to store prototype final coordinates
    * @return a TSS with the added columns planned to process
    */
  def addSOM(predOutColName: String, sourceColName: String, height: Int, width: Int, maxIter: Int,
             centersOutFile: Option[File] = None) = {
    val som = new SOM()
      .setFeaturesCol(sourceColName)
      .setPredictionCol(predOutColName)
      .setMaxIter(maxIter)
      .setHeight(height)
      .setWidth(width)
    val model = som.fit(series)
    centersOutFile.map(f => {
      val writer = new BufferedWriter(new FileWriter(f))
      model.prototypes.zipWithIndex.foreach{case (v, i) => {
        writer.write(i + "," + v.toArray.mkString(",") + "\n")
      }}
      writer.close()
    })
    TSS(model.transform(series))
  }

  def addAutoSOM(predOutColName: String, sourceColName: String, heightBounds: (Int, Int), widthBounds: (Int, Int), autoInitIter: Int, autoIter: Int, maxSOMIter: Int, runsNb: Int = 1, centersOutFile: Option[File] = None)(implicit ss: SparkSession) = {
    def target(vector: DenseVector[Double]) = {
      val height = vector(0)
      val width = vector(0)
      val randId = UUID.randomUUID().toString
      val tempTSS =
        addSOM(predOutColName + randId, sourceColName, height.round.toInt, width.round.toInt, maxSOMIter, None)
          .addSeqFromMLVector(sourceColName + randId, sourceColName)
      val db = tempTSS.daviesBouldin(sourceColName + randId, predOutColName + randId)
      -db
    }
    ???
    // val tuner = new BayesianTuner(target, Array((heightBounds._1, heightBounds._2, "discrete"), (widthBounds._1, widthBounds._2, "discrete")), autoIter, autoInitIter)
    // val optimResult = tuner.tune()
    // val bestHeight = optimResult.x(0)
    // val bestWidth = optimResult.x(0)
    // addSOM(predOutColName, sourceColName, bestHeight.round.toInt, bestWidth.round.toInt, maxSOMIter, centersOutFile)
    ???
  }

  /**
    * Adds a column obtained by applying a UDF to each row
    * @param outColName the added column name
    * @param sourceColNames a Seq[String] of column names, to be used as UDF input
    * @param udf the UDF function to call on sourceColNames
    * @return a TSS with extra column planned to process
    */
  def addUDFColumn(outColName: String, sourceColNames: Seq[String], udf: UserDefinedFunction): TSS = {
    TSS(series.withColumn(outColName, udf(sourceColNames.map(series(_)): _*)))
  }

  /**
    * Adds a column obtained by applying a UDF to each row
    * @param outColName the added column name
    * @param sourceColName a single column name, to be used as UDF input
    * @param udf the UDF function to call on sourceColName
    * @return a TSS with extra column planned to process
    */
  def addUDFColumn(outColName: String, sourceColName: String, udf: UserDefinedFunction): TSS =
    addUDFColumn(outColName, Array(sourceColName), udf)

  /**
    * Adds a column containing a same constant for every row
    * @param outColName the added column name
    * @param source the literal to be contained in every row of the new column
    * @return a TSS with extra column planned to process
    */
  def addConstant[T](outColName: String, source: T) = {
    TSS(series.withColumn(outColName, lit(source)))
  }

  /**
    * Collect a Seq[Double] column as an Array[Array[Double]]
    * @param seqColName the column name to collect
    * @return an Array[Array[Double]] with collected column
    */
  def getSeqColAsArrayDataset(seqColName: String) = {
    series.select(esc(seqColName)).map(_.getSeq[Double](0).toArray).collect()
  }

  /**
    * Adds a Seq[Double] column obtained by setting some values of a Seq[Double] input column to 0 based on a
    * predicate function whitelisting which indices to filter out.
    * @param outColName the added column name.
    * @param sourceColName the input column name from which filtering some values to 0.
    * @param filterFun the predicate function returning true for every index of input column values to set to 0.
    * @return a TSS with added column planned to process.
    */
  def addNullifyValues(outColName: String, sourceColName: String, filterFun: Int => Boolean) = {
    TSS(series.withColumn(outColName, nullifyValuesUDF(filterFun)(series(sourceColName))))
  }

  /**
    * Adds a Seq[Double] column obtained by setting some values of a Seq[Double] input column to 0 based on a
    * user defined range.
    * @param outColName the added column name.
    * @param sourceColName the input column name from which filtering some values to 0.
    * @param startIndex the first index to beginning the value set to 0
    * @param quantity the number of series values to set to 0 from the <startIndex>
    * @return a TSS with added column planned to process.
    */
  def addNullifyValues(outColName: String, sourceColName: String, startIndex: Int, quantity: Int) = {
    TSS(series.withColumn(outColName, nullifyValuesSliceUDF(startIndex, quantity)(series(sourceColName))))
  }

  /**
    * Filter rows based on the existence of some decorator in some specific value.
    * @param decoratorName the decorator name to filter on.
    * @param decoratorValue the decorator value required for decoratorName to remain after filtering.
    * @return the filtered TSS
    */
  def filterDecorator(decoratorName: String, decoratorValue: String) = {
    TSS(series.where(element_at(series(TSS.DECORATORS_COLNAME), decoratorName) === decoratorValue))
  }

  /**
    * Append warnings information of "this" time series as new decorators
    * @param warningsDF the DataFrame containing warnings to append with "scenario_id" column available for join condition.
    * @param ss the SparkSession
    * @return a TSS with decorators extension from warnings planned to process
    */
  def addWarningDecorators(warningsDF: DataFrame)(implicit ss: SparkSession) = {
    //Prepare simulation id decorator as new column on series for allowing join
    val preparedSeries = series.withColumn(TSS.SIMULATIONID_DECORATORNAME, element_at(series(TSS.DECORATORS_COLNAME), TSS.SIMULATIONID_DECORATORNAME))
    //Prepare warnings DF by grouping on simulation id
    val preparedWarningsDF = warningsDF.groupBy("scenario_id").agg(
      first(warningsDF("scenario_name")).alias("scenario_name"),
      collect_list(concat_ws("::", warningsDF("warning_name"), warningsDF("indicator_name"), warningsDF("value"))).alias("warnings")
    )
    //Only load warnings for simulation ids with at least one series in plan
    val joinedDF = preparedSeries.join(preparedWarningsDF,
      preparedSeries(TSS.SIMULATIONID_DECORATORNAME) === preparedWarningsDF("scenario_id"), "left_outer")
    //Encapsulate as new decorators
    TSS(joinedDF.select(
      //All columns of original Dataframe (without new "warnings", join decorator name, and old "decorators" column)
      joinedDF.columns.filterNot(x => {
        x.equals(TSS.SIMULATIONID_DECORATORNAME) || x.equals(TSS.DECORATORS_COLNAME) ||
          x.equals("warnings") || x.equals("scenario_id") || x.equals("scenario_name") || x.equals("decorators")
      }).map(joinedDF(_)) ++
      //New decorators column
      List(
        map_concat(
          map(lit(TSS.SIMULATIONNAME_DECORATORNAME), joinedDF("scenario_name")),
          concatMapWarningsUDF(joinedDF("decorators"), joinedDF("warnings"))
        ).alias("decorators")
      ): _*
    ))
  }

  /**
    * Append metadata information of "this" time series as new decorators
    * @param metadataDF the DataFrame containing metadata to append with <idColName> column available for join condition.
    *                   Numbers decimal separator must be "."
    * @param tssJoinIdColumn the join Column used on <this> side for equality join condition
    * @param metadataDFJoinIdColumn the join Column used on <metadataDF> side for equality join condition
    * @param metadataNames the sequence of <metadataDF> column names to actually add to <this> decorators
    * @param ss the SparkSession
    * @return a TSS with decorators extension from metadata planned to process
    */
  def addMetadataDecorators(metadataDF: DataFrame, tssJoinIdColumn: Column, metadataDFJoinIdColumn: Column, metadataNames: Seq[String])(implicit ss: SparkSession) = {
    //Prepare Metadata DF by transforming its columns into one map column
    val preparedMetadataDF = metadataDF.select(
      metadataDFJoinIdColumn.alias(TSS.SIMULATIONNAME_DECORATORNAME),
      map(metadataNames.flatMap(n => {
        Array(lit(n.trim), metadataDF(n))
      }).toSeq: _*).alias(TSS.METADATA_COLNAME)
    ).persist
    preparedMetadataDF.take(1) //Trigger Partitioning
    //Only load warnings for simulation ids with at least one series in plan
    val joinedDF = addByLeftJoin(preparedMetadataDF, tssJoinIdColumn === preparedMetadataDF(TSS.SIMULATIONNAME_DECORATORNAME))
    //Encapsulate as new decorators
    TSS(joinedDF.series.withColumn(TSS.DECORATORS_COLNAME, map_concat(col(TSS.DECORATORS_COLNAME), col(TSS.METADATA_COLNAME))).drop(TSS.METADATA_COLNAME).drop(TSS.SIMULATIONNAME_DECORATORNAME))
  }

  /**
    * Append metadata information of "this" time series as new decorators
    * @param metadataDF the DataFrame containing metadata to append with <idColName> column available for join condition.
    *                   Numbers decimal separator must be "."
    * @param idColName the <metadataDF> id column name used for join operation
    * @param ss the SparkSession
    * @deprecated Very specific to some use cases. Use addMetadataDecorators(metadataDF: DataFrame, tssJoinIdColumn: Column, metadataDFJoinIdColumn: Column, metadataNames: Seq[String]) for more flexibility.
    * @return a TSS with decorators extension from metadata planned to process
    */
  def addMetadataDecorators(metadataDF: DataFrame, idColName: String = "Test Case ")(implicit ss: SparkSession) = {
    //TODO: Remove dependency with "##" string
    val metadataNames = metadataDF.columns.filterNot(_.equals(idColName))

    //Prepare simulation id decorator as new column on series for allowing join
    val preparedSeries = series.withColumn(TSS.SIMULATIONNAME_DECORATORNAME, element_at(series(TSS.DECORATORS_COLNAME), TSS.SIMULATIONNAME_DECORATORNAME))
    val preparedMetadataDF = metadataDF.select(
      metadataDF(idColName).alias(TSS.SIMULATIONNAME_DECORATORNAME),
      concat_ws("##", metadataNames.map(n => {
        concat_ws("::", lit(n), metadataDF(n)).alias(n)
      }).toSeq: _*).alias(TSS.METADATA_COLNAME)
    )
    //Only load warnings for simulation ids with at least one series in plan
    val joinedDF = preparedSeries.join(preparedMetadataDF,
      preparedSeries(TSS.SIMULATIONNAME_DECORATORNAME) === preparedMetadataDF(TSS.SIMULATIONNAME_DECORATORNAME), "left_outer"
    )
    //Encapsulate as new decorators
    TSS(joinedDF.select(
      //All columns of original Dataframe (without new "warnings", join decorator name, and old "decorators" column)
      joinedDF.columns.filterNot(x => {
        x.equals(TSS.SIMULATIONNAME_DECORATORNAME) || x.equals(TSS.DECORATORS_COLNAME) ||
          x.equals(TSS.METADATA_COLNAME) || x.equals("scenario_name") || x.equals(TSS.DECORATORS_COLNAME)
      }).map(joinedDF(_)) ++
      //New decorators column
      List(concatMapMetadataUDF(joinedDF(TSS.DECORATORS_COLNAME), joinedDF(TSS.METADATA_COLNAME)).alias(TSS.DECORATORS_COLNAME)): _*
    ))
  }

  /**
    * Retrieves the domain union of a map column
    * @param colName the Map[String, Any] column to scan
    * @return the Seq[String] of distinct Map column keys over the whole DataFrame
    */
  def getMapColumnKeys(colName: String) = {
    series.select(colName).rdd.map(_.getMap[String, Any](0).keySet).reduce(_ ++ _).toSeq
  }

  /**
    * Retrieves the decorators domain
    * @return the sorted Seq[String] of distinct decorator names over the whole DataFrame
    */
  def getDecoratorNames = {
    getMapColumnKeys(TSS.DECORATORS_COLNAME).sorted
  }

  /**
    * Map rows to keys using a specific key function, then aggregate results by key using a specific aggregator function, starting from specific "zero" value
    * @param keyFun the function returning a key for a given Row of the DataFrame
    * @param zeroValue the neutral "zero" value with respect to the aggregation function to process
    * @param seqOp the aggregation binary function operating on a previous aggregate result and a single Row
    * @param combOp the aggregation binary function operating on two previous aggregate results
    * @tparam K the Key type for partitioning
    * @tparam U the aggregation type
    * @return a Pair RDD[K, U] with the aggregation result by key
    */
  def aggregateByKey[K, U](keyFun: Row => K, zeroValue: U, seqOp: (U, Row) => U, combOp: (U, U) => U)(implicit kt: ClassTag[K], ut: ClassTag[U], ord: Ordering[K] = null) = {
    series.rdd.map(x => (keyFun(x), x)).aggregateByKey(zeroValue)(seqOp, combOp)
  }

  /**
    * Map rows to keys using a specific key function, then aggregate results by key using a specific aggregator function, starting from specific "zero" value
    * @param keyFun the function returning a key for a given Row of the DataFrame
    * @param zeroValue the neutral "zero" value with respect to the aggregation function to process
    * @param seqOp the aggregation binary function operating on a previous aggregate result and a single Row
    * @param combOp the aggregation binary function operating on two previous aggregate results
    * @param endOp the final map operation to perform from aggregate results
    * @tparam K the Key type for partitioning
    * @tparam U the aggregation type
    * @tparam V the final map output type
    * @return a Pair RDD[(K, V)] with the composition of aggregation then map result by key
    */
  def aggregateByKeyThenMap[K, U, V](keyFun: Row => K, zeroValue: U, seqOp: (U, Row) => U, combOp: (U, U) => U, endOp: U => V)(implicit kt: ClassTag[K], ut: ClassTag[U], vt: ClassTag[V], ord: Ordering[K] = null) = {
    val seriesRDD = series.rdd
    val seriesPairRDD = seriesRDD.map(x => (keyFun(x), x))
    val aggregatedRDD = seriesPairRDD.aggregateByKey(zeroValue)(seqOp, combOp)
    aggregatedRDD.mapValues(endOp)
  }

  /**
    * Retrieve the mean value for a given scalar column, partitioned by key given some key function.
    * @note High computational cost
    * @param keyFun the function returning a key for a given Row of the DataFrame
    * @param meanColName the Double column name on which performing the mean aggregation
    * @tparam K the Key type
    * @return a Pair RDD[(K, Double)] containing the mean by key
    */
  def meanByKey[K](keyFun: Row => K, meanColName: String)(implicit kt: ClassTag[K], ord: Ordering[K] = null) = {
    aggregateByKeyThenMap[K, (Double, Double), Double](keyFun, (0, 0), (p, r) => {
      (p._1 + r.getAs[Double](meanColName), p._2 + 1)
    }, (p1, p2) => (p1._1 + p2._1, p1._2 + p2._2), x => x._1 / x._2)
  }

  /**
    * Retrieve the mean of a given scalar column, by decorator names joint value
    * @param decoratorNames the sequence of decorator names for which means are requested
    * @param meanColName the Double column name on which performing the mean aggregation
    * @return a Dataset[(Seq[String], Double)] associating the requested mean for each value in the cartesian product
    *         of <decoratorNames> domains.
    */
  def meanByDecoratorKey(decoratorNames: Seq[String], meanColName: String) = {
    //TODO: auto detect decorator vs. columns
    val meanColId = decoratorNames.size
    select(decoratorNames ++ List(meanColName, TSS.DECORATORS_COLNAME)).mapDecoratorGroups(decoratorNames: _*){
      case (s, r) => (s, {
        val values = r.map(_.getDouble(meanColId)).toSeq
        values.sum / values.length
      })
    }
  }

  /**
    * Shows rows on as an ASCII table.
    * @note Proxy helper of DataFrame class
    * @param num the number of rows to show
    */
  def show(num: Int) = series.show(num)

  /**
    * Retrieves the mean of a Double column
    * @param colName the column name to aggregate
    * @return the column mean
    */
  def colMean(colName: String) = {
    series.select(mean(series(colName))).collect()(0).getAs[Double](0)
  }

  /**
    * Retrieves the standard deviation of a Double column
    * @param colName the column name to aggregate
    * @return the column standard deviation
    */
  def colSD(colName: String) = {
    series.select(stddev(series(colName))).collect()(0).getAs[Double](0)
  }

  /**
    * Retrieves some quantiles of a Double column
    * @param colName the column name to aggregate
    * @param quantiles the quantile values (in [0, 1])
    * @return the column quantile values
    */
  def colQuantiles(colName: String, quantiles: Array[Double] = Array(0.25, 0.5, 0.75), error: Double = 0.0) = {
    series.stat.approxQuantile(colName, quantiles, error)
  }

  /**
    * Retrieves the gap between the two first consecutive values of a Seq[Double] column.
    * A typical use case is for constant time granularity step inference from a times sequence.
    * Retrieving the information from the first row allows for quick performances.
    * @param colName the Seq[Double] column name
    * @return the column inferred step between the two first sequence values of the first row
    */
  def colSeqFirstStep(colName: String) = {
    series.select(colName).map(row => {
      val vals = row.getSeq[Double](0)
      vals(1) - vals(0)
    })
  }

  /**
    * Retrieves the size of a Seq[Double] column row
    * @param colName the Seq[Double] column name
    * @return the size of the "first" row sequence size for given column name.
    */
  def colSeqSizeOne(colName: String) = {
    series.select(colName).map(_.getSeq[Double](0).size).first
  }

  /**
    * Map Row groups, partitioned by decorator values, using a custom function.
    * @param decoratorNames the decorator names to use for grouping.
    * @param mapFun the function associating a sequence of decorator values and an according rows group to an aggregation value
    * @tparam U the aggregation output type
    * @note key must be manually added to U in <mapFun> if it is important to keep the association from keys to U values
    * @return a Dataset[U] with aggregation results
    */
  def mapDecoratorGroups[U](decoratorNames: String*)(mapFun: (Seq[String], Iterator[Row]) => U)(implicit encoder: Encoder[U]) = {
    groupByDecorators(decoratorNames: _*).mapGroups(mapFun)
  }

  /**
    * Retrieves the domain of a specific decorator name
    * @param decoratorName the decorator to scan
    * @return the decorator domain obtained by the union of all values in all rows for this decorator
    */
  def getDecoratorDomain(decoratorName: String) = {
    getDecoratorDomainRDD(decoratorName).collect
  }

  /**
    * Retrieves a decorator as a dedicated Column
    * @param decoratorName the decorator name to isolate
    * @return the Column object for the isolation operation
    */
  def getDecoratorColumn(decoratorName: String): Column = {
    element_at(series(TSS.DECORATORS_COLNAME), decoratorName)
  }

  /**
    * Select a subset of TSS columns, by names
    * @param colName the first column name
    * @param colNames (optional) the after first column names
    * @return the TSS obtained by keeping the selected columns only
    */
  def select(colName: String, colNames: String*): TSS = {
    select(List(colName) ++ colNames)
  }

  /**
    * Select a subset of TSS columns, by names
    * @param cols the column names to keep
    * @return the TSS obtained by keeping the selected columns only
    */
  def select(cols: Seq[String]): TSS = {
    val columns: Seq[Column] = cols.map(c => {
      if(series.columns.contains(unesc(c))) series(esc(c))
      else getDecoratorColumn(unesc(c))
    })
    selectCols(columns.head, columns.tail: _*)
  }

  /**
    * Remove a subset of columns
    * @param cols the column names to remove
    * @return the TSS obtained by removing the selected columns only
    */
  def drop(cols: String*): TSS = {
    TSS(series.drop(cols: _*))
  }

  /**
    * Select a subset of TSS columns, by Column definition
    * @param column the first Column to select
    * @param columns (optional) the after first Column to select
    * @return the TSS obtained by Column objects only
    */
  def selectCols(column: Column, columns: Column*): TSS = {
    TSS(series.select(List(column) ++ columns: _*))
  }

  /**
    * Select a subset of TSS columns, by Column definition
    * @param columns the Column objects to use for select
    * @return the TSS obtained by Column objects only
    */
  def selectCols(columns: Seq[Column]): TSS = {
    TSS(series.select(columns: _*))
  }

  /**
    * Order the rows by a specific column name
    * @param colName the sort column name
    * @return the TSS obtained by sorting by specific column
    */
  def orderBy(colName: String) = {
    TSS(series.orderBy(colName))
  }

  /**
    * Filter rows by column.
    * Proxy function of DataFrame.where
    * @param condition the filtering condition as a Column object
    * @return the TSS obtained by filtering by <condition>
    */
  def where(condition: Column) = {
    TSS(series.where(condition))
  }

  /**
    * Filter rows by column.
    * Proxy function of DataFrame.where
    * @param conditionStr the filtering condition as a String
    * @return the TSS obtained by filtering by <condition>
    */
  def where(conditionStr: String) = {
    TSS(series.where(conditionStr))
  }

  /**
    * Persist the inner DataFrame.
    * Proxy function of DataFrame.
    * @return this
    */
  def persist() = {
    series.persist()
    this
  }

  /**
    * Unpersist the inner DataFrame.
    * Proxy function of DataFrame.
    * @return this
    */
  def unpersist() = {
    series.unpersist()
    this
  }

  /**
    * Collect the inner DataFrame.
    * Proxy function of DataFrame.
    * @return the collected Array[Row]
    */
  def collect() = series.collect()

  /**
    * Repartition the inner DataFrame
    * Proxy function of DataFrame.
    * @param partitionsNumber the number of partitions after partitioning
    * @return the TSS after repartitioning
    */
  def repartition(partitionsNumber: Int) = {
    TSS(series.repartition(partitionsNumber))
  }

  /**
    * Repartition the inner DataFrame
    * Proxy function of DataFrame.
    * @param partitionsNumber the number of partitions after partitioning
    * @param cols the columns to partition on
    * @return the TSS after repartitioning
    */
  def repartition(partitionsNumber: Int, cols: Column*) = {
    TSS(series.repartition(partitionsNumber, cols: _*))
  }

  /**
    * Coalesces the inner DataFrame
    * Proxy function of DataFrame.
    * @param partitionsNumber the number of partitions after partitioning
    * @return the TSS after repartitioning
    */
  def coalesce(partitionsNumber: Int) = {
    TSS(series.coalesce(partitionsNumber))
  }

  /**
    * Limit the number of rows.
    * Proxy function of DataFrame.
    * @param n the number of rows to keep
    * @return the TSS after limiting its number of rows
    */
  def limit(n: Int) = TSS(series.limit(n))

  /**
    * Sample rows randomly
    * @param f the fraction in [0; 1] of rows to sample
    * @return the TSS after limiting its rows to sampled ones
    */
  def sample(f: Double) = TSS(series.sample(f))

  /**
    * Take n rows from TSS
    * @param n the number of rows
    * @return the n rows as an Array[Row]
    */
  def take(n: Int) = series.take(n)

  /**
    * Retrieve the list of column names
    * @return the column names as an Array[String]
    */
  def columns = series.columns

  /**
    * Retrieve the TSS underlying schema
    * @return the TSS schema as a StructType
    */
  def schema = series.schema

  /**
    * Rename a column
    * @param existingName the name of an existing column to rename
    * @param newName the new name the column to rename to
    * @return the TSS with column renamed
    */
  def withColumnRenamed(existingName: String, newName: String) = TSS(series.withColumnRenamed(existingName, newName))

  def sortByFirstUDF(schema: DataType) = udf((rows: Seq[Row]) => {
    rows.sortBy(_.getString(0))//.map(_.getInt(1))//.toSeq.drop(1)
  }, schema)

  def arrayStructToStructArray(schema: StructType) = udf((rows: Seq[Row]) => {
    Row(schema.fields.indices.map(i => rows.map(_.get(i))): _*)
  }, schema)

  def group(key: Column, inKeyOrder: Column) = {
    val columns = Seq(inKeyOrder.alias(TSS.INGROUPKEY_COLNAME)) ++ series.columns.map(c => series(c))
    val colNames = series.select(columns: _*).columns
    val structCol = collect_list(struct(columns: _*))
    val structSchema = selectCols(structCol).series.schema(0).dataType
    val fields = structSchema.asInstanceOf[ArrayType].elementType.asInstanceOf[StructType].fields
    val finalSchema = StructType((0 until fields.length).map(i => {
      StructField(colNames(i), ArrayType(fields(i).dataType))
    }).toArray)
    val tempDF = series
      .groupBy(key.alias(TSS.KEY_COLNAME))
      .agg(sortByFirstUDF(structSchema)(structCol).alias("toexplode"))
    val groupedDF = tempDF.select(tempDF(TSS.KEY_COLNAME), arrayStructToStructArray(finalSchema)(tempDF("toexplode")).alias("toexplode"))
    TSS(groupedDF.select(TSS.KEY_COLNAME, "toexplode.*"), true)
  }

  /**
    * Unify two TSS objects and recompute unique identifiers for resulting rows.
    * Proxy function of DataFrame, with extra unique id recomputation.
    * @param other the TSS to unify with "this"
    * @return the TSS after unification
    */
  def union(other: TSS) = {
    val unifiedSeries = series.union(other.series)
    TSS(unifiedSeries.drop(TSS.ID_COLNAME).withColumn(TSS.ID_COLNAME, monotonically_increasing_id()))
  }

  /**
    * Recompute unique identifiers for rows.
    * @return the TSS after reindexing rows
    */
  def regenerateIndices = {
    TSS(series.drop(TSS.ID_COLNAME).withColumn(TSS.ID_COLNAME, monotonically_increasing_id()))
  }

  /**
    * Helper for selecting columns like in DataFrame API, with parenthesis.
    * @param colName the column name to select
    * @return the Column object corresponding to the column name
    */
  def apply(colName: String) = series(colName)

  /**
    * Save a TSS inner DataFrame as a compressed distributed chunked gzipped parquet file
    * @param folderPath the folder path containing the output file chunks
    * @param overwrite whether to allow automatic overwrite of already existing file
    */
  def save(folderPath: String, overwrite: Boolean = false) = {
    val conf = series.write.option("compression", "gzip")
    if(overwrite) conf.mode("overwrite").save(folderPath)
    else conf.save(folderPath)
  }

  /**
    * Save a TSS inner DataFrame as a com.databricks.spark.csv file, after flattening Seq and Map columns.
    * @param path the output file chunks path
    * @param header whether to add header at the beginning of the file
    * @param overwrite whether to allow automatic overwrite of already existing file
    */
  def saveCSV(path: String, header: Boolean = true, overwrite: Boolean = false) = {
    val schema = series.schema
    series.select(schema.flatMap(col => {
      col.dataType match {
        case ArrayType(_, _) => List(concat_ws(";", series(esc(col.name))).alias(col.name))
        case MapType(_, _, _) => {
          val keys = getMapColumnKeys(col.name)
          keys.map(k => getDecoratorColumn(k).alias(TSS.DECORATORS_COLNAME + "_" + k))
        }
        case _ => List(series(esc(col.name)))
      }
    }): _*).write.format("com.databricks.spark.csv").option("header", header).save(path)
  }
  //END INTERFACE FUNCTIONS

}

object TSS {

  val SIMULATIONID_DECORATORNAME: String = "simulationId"
  val SIMULATIONNAME_DECORATORNAME: String = "simulationName"
  val SIMULATIONPLANID_DECORATORNAME: String = "planId"

  val METADATA_COLNAME: String = "metadata"
  val DECORATORS_COLNAME: String = "decorators"
  val TIMEFROM_COLNAME: String = "timeFrom"
  val TIMETO_COLNAME: String = "timeTo"
  val TIMEGRANULARITY_COLNAME: String = "timeGranularity"
  val SERIES_COLNAME: String = "series"
  val ID_COLNAME: String = "id"

  val INGROUPKEY_COLNAME: String = "inGKey"
  val KEY_COLNAME: String = "gkey"

  /**
    * Helper to retrieve the decorator value of a specific decorator of a Row object
    * @param row the Row to look for decorator
    * @param decoratorName the decorator name
    * @return the corresponding <row> value for <decoratorName>
    */
  def getDecorator(row: Row, decoratorName: String) = {
    row.getAs[Map[String, String]]("decorators")(decoratorName)
  }

  /**
    * List all files of a given folder as File objects
    * @param folderPath the root folder path
    * @param includeDirectories whether to include directories in the result
    * @return a List[File] object with an item per file (and optionally directory)
    */
  def getFolderFiles(folderPath: String, includeDirectories: Boolean = false) = {
    val d = new File(folderPath)
    if (d.exists && d.isDirectory) {
      if(includeDirectories)
        d.listFiles.toList
      else
        d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  /**
    * Load a TSS object from a chunked parquet representation of its inner DataFrame
    * @param folderPath the folder path used for saving (the one containing series.txt)
    * @param ss the SparkSession
    * @return a TSS object with loaded inner DataFrame
    */
  def load(folderPath: String)(implicit ss: SparkSession): TSS = {
    TSS(ss.read.load(folderPath))
  }

  /**
    * Transpose a DataFrame object
    * @param df the DataFrame to tranpose
    * @param ss the SparkSession
    * @note only numeric columns are transposed
    * @return the transposed DataFrame object
    */
  def transpose(df: DataFrame)(implicit ss: SparkSession) = {
    val colNames = df.schema.map(_.name)
    val validColNames = df.schema.filter(x => x.dataType.equals(IntegerType) || x.dataType.equals(DoubleType)).map(_.name)
    val doubleDF = df.select(validColNames.map(c => df("`" + c + "`").cast(DoubleType)): _*)
    doubleDF.rdd.flatMap(x => x.toSeq.map(_.asInstanceOf[Double]).zipWithIndex)
      .groupBy(_._2)
      .map(x => (colNames(x._1), x._2.map(_._1)))
  }

  /**
    * Builds a TSS object from a scenario file loaded as a DataFrame
    * @param df the DataFrame with "time" column and a column per variable of a given scenario
    * @param decorators the decorators to add to every input variable column output univariate time series
    * @param timeColumnId the name of input DataFrame time column
    * @param ss the SparkSession
    * @return a TSS object with one row per non time input column, indexed by the time column
    */
  def build(df: DataFrame, decorators: Map[String, String], timeColumnId: String = "time")(implicit ss: SparkSession): TSS = {
    import ss.implicits._
    val times = df.select(timeColumnId).map(_.getDouble(0)).collect().sorted
    val timeFrom = times.head
    val timeTo = times.last
    val timeGranularity = (times.last - times.head) / (times.size - 1.0)
    val timeSeriesRDD = transpose(df)
    TSS(timeSeriesRDD.map(x => (timeFrom, timeTo, timeGranularity, decorators.updated("varName", x._1), x._2.toArray))
      .toDF(TSS.TIMEFROM_COLNAME, TSS.TIMETO_COLNAME, TSS.TIMEGRANULARITY_COLNAME, TSS.DECORATORS_COLNAME, TSS.SERIES_COLNAME))
  }

  /**
    * Builds a TSS object from a scenario file path
    * @param inputPath the scenario file path
    * @param format the file format ("csv", "parquet", or any other format String recognized by SparkSession.read.format)
    * @param decorators the decorators to add to every input variable column output univariate time series
    * @param ss the SparkSession
    * @return a TSS object with one row per non time input column, indexed by the time column
    */
  def readSCANeROne(inputPath: String, format: String, decorators: Map[String, String] = Map())(implicit ss: SparkSession): TSS = {
    import ss.implicits._
    build(ss.read.format(format)
      .option("inferSchema", true)
      .option("delimiter", "\t")
      .option("header", "true")
      .option("nullValue", "null")
      .load(inputPath)
      .where($"time" > 0), decorators)
  }

  /**
    * Builds a TSS object from multiple scenario file paths
    * @param inputPaths the scenario file paths
    * @param format the files format ("csv", "parquet", or any other format String recognized by SparkSession.read.format)
    * @param fileToScenarioId a function mapping a scenario File object to the simulationId decorator value (often in file path)
    * @param decorators the decorators to add to every input variable column output univariate time series
    * @param ss the SparkSession
    * @return a TSS object with one row per non time input column, indexed by the time column
    */
  def readSCANeRMany(inputPaths: Seq[String], format: String, fileToScenarioId: File => String, decorators: Map[String, String] = Map())(implicit ss: SparkSession): TSS = {
    (0 until inputPaths.length).par.map(i => {
      val p = inputPaths(i)
      val f = new File(p)
      val fn = f.getName
      val localDecorators = decorators ++ Map("simulationId" -> fileToScenarioId(f))
      TSS.readSCANeROne(p, format, localDecorators)(ss)
    }).reduce((x, y) => {
      x.union(y)
    })
  }

  /**
    * Builds a TSS object from a scenarios plan folder path, containing multiple scenario files
    * @param inputPath the plan folder path
    * @param format the folder files format ("csv", "parquet", or any other format String recognized by SparkSession.read.format)
    * @param decorators the decorators to add to every input variable column output univariate time series
    * @param ss the SparkSession
    * @return a TSS object with one row per non time input column, indexed by the time column
    */
  def readSCANeRFolder(inputPath: String, format: String, decorators: Map[String, String] = Map(), maxScenariosNumber: Int = Integer.MAX_VALUE)(implicit ss: SparkSession) = {
    val d = new File(inputPath)
    val files = getFolderFiles(inputPath)
    (0 until scala.math.min(maxScenariosNumber, files.length)).par.map(i => {
      val f = files(i)
      val fn = f.getName
      val localDecorators = decorators ++ Map("simulationId" -> fn.replace("_scaner.txt", "").split("_")(1))
      readSCANeROne(f.getAbsolutePath, format, localDecorators)
    }).reduce((x, y) => {
      x.union(y)
    })
    /**/
  }

  /**
    * Build a TSS object from 3 dataframes corresponding to raw data, metadata and warnings
    * @param rawData the raw data DataFrame object, in the |time|var1|var2|...|scenarioId| columns format with single time step per row
    * @param metadata the metadata DataFrame object
    * @param warnings the warnings DataFrame object
    * @param varNames the variable names whitelist to consider to keep in the output TSS API. It must corresponds to <rawData> column names
    * @param varsToProcessPerCall the number of variables to process per pivot batch (reduce this value if out of memory errors occur)
    * @param rawDataScenarioIdColName the rawData scenario id column name
    * @param rawDataPlanIdColName the rawData plan id column name
    * @param rawDataTimeColName the rawData time column name
    * @param metadataScenarioIdColName the metadata scenario id column name (used to join with warnings and rawData)
    * @param warningsScenarioIdColName the warnings scenario id column name (used to join with metadata and rawData)
    * @param warningsColumnNameSeparator the String separating column hierarchical description parts in warning DataFrame (example: WarningXXX.value uses "." to separate members of WarningXXX)
    * @return a TSS object with one row per non time input column, indexed by the time column
    */
  def readSCANeRDataFrames(rawData: DataFrame, metadata: DataFrame, warnings: DataFrame, varNames: Option[Seq[String]], varsToProcessPerCall: Int = 100,
                           rawDataScenarioIdColName: String = "scenarioId", rawDataPlanIdColName: String = "simulationPlanId",
                           rawDataTimeColName: String = "time", metadataScenarioIdColName: String = "_id", warningsScenarioIdColName: String = "scenario_id",
                           warningsColumnNameSeparator: String = ".")(implicit ss: SparkSession) = {

    import ss.implicits._

    //Used to transform posexplode ids into var names
    def indexToValueUDF(varNames: Seq[String]) = udf((index: Int) => varNames(index))

    //Prepare raw data by partitioning by scenario and sorting within each partition by time
    val tt = rawData.repartition(col(rawDataScenarioIdColName)).sortWithinPartitions(rawDataTimeColName).persist

    //Retrieve valid column names: the ones of Double type and part of the white list given as parameter
    val allDoubleColNames = rawData.schema.filter(_.dataType == DoubleType).map(_.name)
    val actualVarNames = varNames.getOrElse(allDoubleColNames.filter(x => x != rawDataTimeColName).toSeq)
    val actualVarsNumber = actualVarNames.size

    //Organize the "pivot" by batch of <varsToProcessPerCall> maximum size, to avoid out of memory issues
    val transformedRawData = ((0 until actualVarsNumber by varsToProcessPerCall).map(i => {

      val iterationColumnUBound = scala.math.min(i + varsToProcessPerCall, actualVarsNumber)
      val iterationActualVarNames = actualVarNames.slice(i, iterationColumnUBound)

      //Group By Mandatory since repartitioning by scenarioId does not guarantee that only one scenarioId will be present per partition
      tt.groupBy(rawDataScenarioIdColName)
        .agg(
          collect_list(col(rawDataTimeColName)).alias("time"), first(rawDataPlanIdColName).alias("planId"),
          array(iterationActualVarNames.map(vn => collect_list(col(vn))): _*).alias("values")
        )
        //Do pivot for the given variable names batch
        .select(
        col(rawDataScenarioIdColName), col("planId"),
        element_at(col("time"), 1).alias("timeFrom"),
        element_at(reverse(col("time")), 1).alias("timeTo"),
        (element_at(col("time"), 2) - element_at(col("time"), 1)).alias("timeGranularity"),
        posexplode_outer(col("values"))
      )
        .select(
          map(lit("simulationId"), col(rawDataScenarioIdColName).alias("simulationId"), lit("planId"), col("planId"),
            lit("varName"), indexToValueUDF(iterationActualVarNames)(col("pos")).alias("varName")).alias("decorators"),
          col("timeFrom"), col("timeTo"), col("timeGranularity"),
          col("col").alias("series")
        )
      //Union all batches together
    }).reduce((df1, df2) => df1.union(df2)))

    val cleanedColumnsWarnings = warnings.columns.filter(!_.equals(warningsScenarioIdColName)).foldLeft(warnings){case (df, c) => {
      df.withColumnRenamed(c, c.replace(warningsColumnNameSeparator, "::"))
    }}

    //transformedRawData.write.mode("overwrite").parquet(planSettings.dbfsOutFolderPath + "/plan_raw_grouped.parquet")
    //val tss = TSS(spark.read.parquet(planSettings.dbfsOutFolderPath + "/plan_raw_grouped.parquet"))
    //Create the TSS object, now the inner data is adequately set
    val tss = TSS(transformedRawData)
    //Join Metadata and Warnings
    //val metadataAndWarningsDF = metadata.join(cleanedColumnsWarnings, metadata("_id") === cleanedColumnsWarnings("scenario_id"), "left_outer").drop(metadata("_id"))
    val metadataAndWarningsDF = metadata.join(cleanedColumnsWarnings, metadata(metadataScenarioIdColName) === cleanedColumnsWarnings(warningsScenarioIdColName), "left_outer").drop(metadata(metadataScenarioIdColName)).persist
    metadataAndWarningsDF.take(1) //Force Partitioning
    val decoratedPlan = tss.addMetadataDecorators(metadataAndWarningsDF, element_at(tss("decorators"), "simulationId"), metadataAndWarningsDF(warningsScenarioIdColName), metadataAndWarningsDF.columns)//.persist
    //decoratedPlan.repartition(planSettings.partitionsNumber).save(planSettings.dbfsOutFolderPath + "/plan_with_decorators.parquet", true)
    //tt.unpersist
    decoratedPlan
  }

  /**
    * Weighted Squared Euclidean distance function between two V objects, V inheriting from Seq[Double]
    * @param weights the numeric weights vector
    * @param v1 the first numeric "vector" object
    * @param v2 the second numeric "vector" object
    * @tparam V the input vector concrete types
    * @return the squared euclidean distance between <v1> and <v2>
    */
  def weightedEuclideanF[V <: Seq[Double]](weights: V)(v1: V, v2: V) = {
    @annotation.tailrec
    def go(d: Double, i: Int): Double = {
      if( i < v1.size ) {
        val toPow2 = v1(i) - v2(i)
        go(d + weights(i) * toPow2 * toPow2, i + 1)
      }
      else d
    }
    //sqrt(
    go(0D, 0)//)
  }

  /**
    * (Squared or not) Euclidean distance function between two V objects, V inheriting from Seq[Double]
    * @param squared whether to return the squared Euclidean distance (avoid extra computation whenever
    *                only the ranking of distances are useful) or true one (if the distance itself is important,
    *                e.g. for an MDS proximity matrix.
    * @param v1 the first numeric "vector" object
    * @param v2 the second numeric "vector" object
    * @tparam V the input vector concrete types
    * @return the (squared) euclidean distance between <v1> and <v2>
    */
  def euclideanF[V <: Seq[Double]](squared: Boolean)(v1: V, v2: V) = {
    @annotation.tailrec
    def go(d: Double, i: Int): Double = {
      if(i < v1.size) {
        val toPow2 = v1(i) - v2(i)
        go(d + toPow2 * toPow2, i + 1)
      }
      else d
    }
    if (squared) go(0D, 0) else scala.math.sqrt(go(0D, 0))
    //sqrt(
    //go(0D, 0)//)
  }

  /**
    * Hamming distance function between two V objects, V inheriting from Seq[Int]
    * @param v1 the first Int "vector" object
    * @param v2 the second Int "vector" object
    * @tparam V the input vector concrete types
    * @return the hamming distance between <v1> and <v2>
    */
  def hammingF[V <: Seq[Int]](v1: V, v2: V) = {
    @annotation.tailrec
    def go(d: Int, i: Int): Double = {
      if (i < v1.size) {
        val hammingPoint = if(v1(i) != v2(i)) 1 else 0
        go(d + hammingPoint, i + 1)
      }
      else d
    }
    go(0, 0)
  }

  /**
    * Compute distances between all vectors of a Dataset
    * @param points the Dataset of k points description as (Long, Seq[A]) pairs where Long is an id and Seq[A] is the collection used for distance computations
    * @return a TSDS object with the distances planned to process
    */
  def distances[A](points: Dataset[(Long, Seq[A])], dist: (Seq[A], Seq[A]) => Double)(implicit ss: SparkSession) = {
    import ss.implicits._
    points.persist()
    points.take(1) //Ensure persist is achieved before join is called
    val crossPoints = points.crossJoin(points)
    val ds = crossPoints.flatMap(x => {
      val id1 = x.getLong(0)
      val id2 = x.getLong(2)
      if(id1 <= id2){
        Some(IndexedDistance(id1, id2, dist(x.getSeq[A](1), x.getSeq[A](3))))
      }else{
        None
      }
    })
    val symDS = ds.select(ds("id2").alias("id1"), ds("id1").alias("id2"), ds("distance")).as[IndexedDistance]
    val fullDS = ds.union(symDS.where(symDS("id1") =!= symDS("id2")))
    TSDS(fullDS, points.count.toInt)
  }

  //ML API Version
  def logLikelihoodIndividualML(model: GaussianMixtureModel) = udf((features: org.apache.spark.ml.linalg.Vector) => {
    val perComponentLogPdfs = (0 until model.gaussians.length).map(i => {
      model.gaussians(i).logpdf(features) + scala.math.log(model.weights(i)) //Check the case where it is 0, with a default value as Etienne
    })
    //scala.math.log(perComponentPdfs.sum)
    Utils.logSumExp(perComponentLogPdfs)
  })

  //MLLib API Version
  def logLikelihoodIndividualMLLib(model: org.apache.spark.mllib.clustering.GaussianMixtureModel) = udf((features: org.apache.spark.ml.linalg.Vector) => {
    val perComponentLogPdfs = (0 until model.gaussians.length).map(i => {
      model.gaussians(i).logpdf(org.apache.spark.mllib.linalg.DenseVector.fromML(features.toDense)) + scala.math.log(model.weights(i)) //Check the case where it is 0, with a default value as Etienne
    })
    Utils.logSumExp(perComponentLogPdfs)
  })

  //MLLib API Version
  def predictUDFMLLib(mlLibModel: org.apache.spark.mllib.clustering.GaussianMixtureModel) = udf((source: org.apache.spark.ml.linalg.Vector) => {
    mlLibModel.predict(org.apache.spark.mllib.linalg.DenseVector.fromML(source.toDense))
  })

  //MLLib API Version
  def softPredictUDFMLLib(mlLibModel: org.apache.spark.mllib.clustering.GaussianMixtureModel) = udf((source: org.apache.spark.ml.linalg.Vector) => {
    mlLibModel.predictSoft(org.apache.spark.mllib.linalg.DenseVector.fromML(source.toDense))
  })


}