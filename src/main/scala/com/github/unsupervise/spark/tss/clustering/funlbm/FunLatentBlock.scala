/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.funlbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import Tools._
import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import breeze.numerics.abs
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer
import scala.util.{Failure, Random, Success, Try}


class FunLatentBlock private(private var K: Int,
                             private var L: Int,
                             var maxIterations: Int,
                             var maxBurninIterations: Int,
                             private var updateLoadings: Boolean = false,
                             private var updateLoadingStrategy: Double = 0,
                             private var fullCovarianceHypothesis: Boolean = true,
                             var maxPcaAxis : Int = 5,
                             private var seed: Long) {

  val precision = 1e-2

  /**
    * Constructs a default instance.
    */
  def this() = this(1, 1 , 20, 20, seed = Random.nextLong())

  // an initializing model can be provided rather than using the
  // default random starting point
  private var providedInitialModel: Option[FunLatentBlockModel] = None

  /**
    * Set the number of Gaussians in the mixture model.  Default: 2
    */
  def setK(K: Int): this.type = {
    require(K > 0,
      s"Number of row clusters must be positive but got $K")
    this.K = K
    this
  }

  /**
    * Set the number of Gaussians in the mixture model.  Default: 2
    */
  def setL(L: Int): this.type = {
    require(L > 0,
      s"Number of column clusters must be positive but got $L")
    this.L = L
    this
  }

  /**
    * Return the number of row cluster number in the latent block model
    */
  def getK: Int = K

  /**
    * Return the number of column cluster number in the latent block model
    */
  def getL: Int = L


  /**
    * Set the maximum pca axis number to retain during M step
    */
  def setMaxPcaAxis(maxPcaAxis: Int): this.type = {
    require(maxPcaAxis > 0)
    this.maxPcaAxis = maxPcaAxis
    this
  }

  /**
    * Set the maximum pca axis number to retain during M step
    */
  def getMaxPcaAxis: Int = maxPcaAxis

  /**
    * Set the maximum number of iterations allowed. Default: 100
    */
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations > 0,
      s"Maximum of iterations must be strictly positive but got $maxIterations")
    this.maxIterations = maxIterations
    this
  }

  /**
    * Set the maximum number of iterations allowed. Default: 100
    */
  def setMaxBurninIterations(maxBurninIterations: Int): this.type = {
    require(maxBurninIterations >= 0,
      s"Maximum of Burn-in iterations must be positive or zero but got $maxBurninIterations")
    this.maxBurninIterations = maxBurninIterations
    this
  }

  /**
    * Set the initial GMM starting point, bypassing the random initialization.
    * You must call setK() prior to calling this method, and the condition
    * (model.K == this.K) must be met; failure will result in an IllegalArgumentException
    */
  def setInitialModel(model: FunLatentBlockModel): this.type = {
    require(model.K == K,
      s"Mismatched row cluster number (model.KVec ${model.K} != KVec $K)")
    providedInitialModel = Some(model)
    this
  }

  /**
    * Set Whether the loadings are updated at each M step, or not
    */
  private def setUpdateLoadings(updateLoadings: Boolean): this.type = {
    this.updateLoadings = updateLoadings
    this
  }

  def setUpdateLoadingStrategy[T](updateLoadingStrategy: T): this.type = {
    val updateLoadingStrategyTmp = updateLoadingStrategy.toString
    require(List("always","never").contains(updateLoadingStrategyTmp) ||
      (isInteger(updateLoadingStrategyTmp) &
        updateLoadingStrategyTmp.toInt >= 0), "updateLoadingStrategy should be an int >= 0 or ('always', 'never'))")

    this.updateLoadingStrategy = updateLoadingStrategyTmp match {
      case "always" => 0D
      case "never" => Double.PositiveInfinity
      case _ => updateLoadingStrategyTmp.toInt.toDouble
    }
    this
  }
  def getUpdateLoadingStrategy: Double = updateLoadingStrategy

  def computeMeanModels(models: List[FunLatentBlockModel]): FunLatentBlockModel = {

    val meanProportionRows: List[Double] = (models.map(model =>
      DenseVector(model.proportionsRows.toArray)).reduce(_+_)/models.length.toDouble).toArray.toList

    val meanProportionCols: List[Double] = (models.map(model =>
      DenseVector(model.proportionsCols.toArray)).reduce(_+_)/models.length.toDouble).toArray.toList

    val meanMu: DenseMatrix[DenseVector[Double]] = DenseMatrix.tabulate(K,L){ (k, l)=>{
      models.map(m => DenseVector(m.gaussians(k)(l).mean.toArray)).reduce(_+_)/models.length.toDouble
    }}

    val meanCovariance: DenseMatrix[DenseMatrix[Double]] = DenseMatrix.tabulate(K,L){ (k, l)=>{
      models.map(m => DenseMatrix(m.gaussians(k)(l).cov.toArray).reshape(meanMu(0,0).length,meanMu(0,0).length)).reduce(_+_)/models.length.toDouble
    }}

    val meanGaussians: List[List[MultivariateGaussian]] = (0 until K).map(k => {
      (0 until L).map(l => {
        new MultivariateGaussian(
          Vectors.dense(meanMu(k,l).toArray),
          denseMatrixToMatrix(meanCovariance(k,l)
          ))
      }).toList
    }).toList

    new FunLatentBlockModel(meanProportionRows, meanProportionCols, meanGaussians, models.head.loadings, models.head.centerings)
  }

  def run(periodogram: RDD[(Int, Array[DenseVector[Double]])],
          EMMethod: String= "SEMGibbs",
          verbose: Boolean = false,
          nConcurrent:Int=5,
          nTryMaxPerConcurrent:Int=10,
          initMethod: String = "randomPartition")(implicit ss: SparkSession): Map[String,Product] = {

    val n:Int = periodogram.count().toInt
    val p:Int = periodogram.take(1).head._2.length

    val allRes = (0 until nConcurrent).map(nTry => {
      if(verbose){println("> n° launch "+(1+nTry).toString+"/"+nConcurrent.toString)}
      this.initAndRunTry(periodogram, n, p, EMMethod, nTry =1, nTryMax = nTryMaxPerConcurrent, initMethod = initMethod, verbose=verbose)
    }).toList
    val allLikelihoods: DenseVector[Double] = DenseVector(allRes.map(_("LogLikelihood").asInstanceOf[List[Double]].last).toArray)

    allRes(argmax(allLikelihoods))
  }


  def initAndRunTry(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                    n: Int,
                    p: Int,
                    EMMethod:String,
                    nTry: Int,
                    nTryMax: Int,
                    initMethod: String,
                    verbose: Boolean)(implicit ss: SparkSession): Map[String,Product] = {

    val t0 = System.nanoTime()
    val logLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity
    val ICL: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity

    if(nTry > nTryMax){
      return Map("Model" -> new FunLatentBlockModel(),
        "RowPartition" -> List.fill(n)(0),
        "ColPartition" -> List.fill(p)(0),
        "LogLikelihood" -> logLikelihoodList.toList,
        "ICL" -> ICL.toList)}

    Try(this.initAndLaunch(periodogram, n,p,EMMethod, initMethod = initMethod, verbose=verbose)) match {
      case Success(v) =>
        if(verbose){println()
          printTime(t0, EMMethod+" FunLBM Spark")}
        Success(v).get.asInstanceOf[Map[String, Product with Serializable]]
      case Failure(e) =>
        if(verbose){
          println("Failure: " + e)
          if(nTry==1){
            print("Algorithm "+ EMMethod+" didn't converge to an appropriate solution, trying again..\n" +
              "n° try: "+nTry.toString+"")
          } else {print(", "+nTry.toString)}}
        this.initAndRunTry(periodogram, n, p, EMMethod, nTry+1,nTryMax, initMethod=initMethod, verbose=verbose)
    }
  }

  def initAndLaunch(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                    n:Int,
                    p:Int,
                    EMMethod: String,
                    verbose:Boolean,
                    initMethod: String)(implicit ss: SparkSession): Map[String,Product]= {

    val (initialModel, initialColPartition) = providedInitialModel match {
      case Some(model) =>
        require(initMethod.isEmpty,
          s"An initial model has been provided but initMethod argument has also been set to $initMethod. " +
            s"Please make a choice: do not set an initMethod or do not provide an initial model.")
        (model, (0 until p).map(_ => sample(model.proportionsCols)).toList)
      case None => Initialization.initialize(periodogram,this,n,p,
        this.fullCovarianceHypothesis,this.maxPcaAxis, verbose,initMethod)
    }
    if(verbose) println("Initialization done")
    launchEM(periodogram, EMMethod, initialColPartition,initialModel,n, p,verbose)
  }

  def launchEM(periodogram: RDD[(Int, Array[DenseVector[Double]])],
               EMMethod: String,
               initialColPartition: List[Int],
               initialModel: FunLatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean)(implicit ss: SparkSession): Map[String,Product] = {

    require(List("SEMGibbs","VariationalEM").contains(EMMethod),
      "EM Method provided "+EMMethod+" is not available. Possible options: SEMGibbs")
    EMMethod match {
      case "SEMGibbs" => SEMGibbs(periodogram, initialColPartition, initialModel,n,p, verbose)
    }
  }

  def SEMGibbs(periodogram: RDD[(Int, Array[DenseVector[Double]])],
               initialColPartition: List[Int],
               initialModel: FunLatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean)(implicit ss: SparkSession): Map[String,Product] = {
    println("SEMGibbs begins")
    require(this.fullCovarianceHypothesis, "in SEM-Gibbs mode, indep. covariance hyp. is not yet available," +
      " please set latentBlock.fullCovarianceHypothesis to true")
    this.setUpdateLoadings(false)
    var precPeriodogram: RDD[(Int, Array[DenseVector[Double]], Int)] = periodogram.map(r => (r._1, r._2, 0))
    var precColPartition = initialColPartition
    precPeriodogram = initialModel.drawRowPartition(precPeriodogram, precColPartition)
    var precModel   = initialModel
    if(verbose){precPeriodogram = precModel.SEMGibbsExpectationStep(precPeriodogram, precColPartition, verbose = verbose)._1}

    var completeLogLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]() :+ Double.NegativeInfinity :+
      precModel.completelogLikelihood(
        projectInLowerSubspace(precPeriodogram, precColPartition, precModel.loadings, precModel.centerings),
        precColPartition)

    if(verbose){println(">>> Initial model")}
    if(verbose){println("Loglikelihood: " + completeLogLikelihoodList.last.toString)}

    var cachedModels = List[FunLatentBlockModel]() :+ precModel
    var hasConverged: Boolean = false
    var t0 = System.nanoTime
    var iter = 0
    do {
      iter += 1
      if(verbose){println(">>> iter: "+iter.toString)}
      if(iter>this.updateLoadingStrategy){this.setUpdateLoadings(true)}
      if(verbose){println("Update loading: "+this.updateLoadings.toString)}
      val (newData, newColPartition) = precModel.SEMGibbsExpectationStep(precPeriodogram, precColPartition, verbose = verbose)
      if(verbose){t0 = printTime(t0, "SE")}
      val newModel = precModel.SEMGibbsMaximizationStep(newData, newColPartition,
        this.fullCovarianceHypothesis, this.updateLoadings, this.maxPcaAxis, verbose)
      if(verbose){t0 = printTime(t0, "Maximization")}
      if(verbose){println("New Row Partitions:" + newData.map(row => (row._1, row._3)).collect().sortBy(_._1).map(_._2).toList.mkString(","))}
      precModel = newModel
      precPeriodogram = newData
      precColPartition = newColPartition
      cachedModels = cachedModels :+ precModel
      completeLogLikelihoodList += precModel.completelogLikelihood(
        projectInLowerSubspace(precPeriodogram, precColPartition, precModel.loadings, precModel.centerings), precColPartition)
      if(verbose){println("Loglikelihood: " + completeLogLikelihoodList.last.toString)}

      hasConverged = abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(1).last)<precision &
        abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(2).last)<precision

    } while (iter < (maxBurninIterations+maxIterations) & !hasConverged)

    val iclList = completeLogLikelihoodList.map(ll => precModel.ICL(ll,n,p,true)).toList.drop(1)

    val resModel = if (!hasConverged){
      computeMeanModels(cachedModels.drop(1+maxBurninIterations))
    } else cachedModels.last

    val rowPartition: List[Int] = precPeriodogram.map(row => (row._1, row._3)).collect().sortBy(_._1).map(_._2).toList

    Map("Model" -> resModel,
      "RowPartition" -> rowPartition,
      "ColPartition" -> precColPartition,
      "LogLikelihood" -> completeLogLikelihoodList.toList.drop(1),
      "ICL" -> iclList.drop(1))

  }

}
