/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.lbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ToolsRDD._
import breeze.linalg.{*, DenseMatrix, DenseVector, argmax}
import breeze.numerics.abs
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer
import scala.util.{Failure, Random, Success, Try}


class LatentBlock private(private var K: Int,
                          private var L: Int,
                          private var maxIterations: Int,
                          private var maxBurninIterations: Int,
                          private var fullCovarianceHypothesis: Boolean = true,
                          private var seed: Long) extends Serializable {

  val precision = 1e-2

  /**
    * Constructs a default instance. The default parameters are {k: 2, convergenceTol: 0.01,
    * maxIterations: 100, seed: random}.
    */
  def this() = this(2,2, 10, 10, seed = Random.nextLong())

  // number of samples per cluster to use when initializing Gaussians
  private val nSamples = 5

  // an initializing model can be provided rather than using the
  // default random starting point
  private var providedInitialModel: Option[LatentBlockModel] = None

  /**
    * Set the initial GMM starting point, bypassing the random initialization.
    * You must call setK() prior to calling this method, and the condition
    * (model.k == this.k) must be met; failure will result in an IllegalArgumentException
    */
  def setInitialModel(model: LatentBlockModel): this.type = {
    require( model.K == K,
      s"Mismatched row cluster number (model.K ${model.K} != K $K)")
    require( model.L == L,
      s"Mismatched column cluster number (model.L ${model.L} != L $L)")
    providedInitialModel = Some(model)
    this
  }

  /**
    * Return the user supplied initial GMM, if supplied
    */
  def getInitialModel: Option[LatentBlockModel] = providedInitialModel

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
    * Return the maximum number of iterations allowed
    */
  def getMaxIterations: Int = maxIterations

  /**
    * Return the maximum number of iterations allowed
    */
  def getMaxBurninIterations: Int = maxBurninIterations

  /**
    * Set the number of Gaussians in the mixture model.  Default: 2
    */
  def setFullCovarianceHypothesis(fullCovarianceHypothesis: Boolean): this.type = {
    this.fullCovarianceHypothesis = fullCovarianceHypothesis
    this
  }

  /**
    * Return the number of row cluster number in the latent block model
    */
  def getFullCovarianceHypothesis: Boolean = fullCovarianceHypothesis

  /**
    * Set the random seed
    */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
    * Return the random seed
    */
  def getSeed: Long = seed

  def computeMeanModels(models: List[LatentBlockModel]): LatentBlockModel = {

    val meanProportionRows: List[Double] = (models.map(model =>
      DenseVector(model.proportionsRows.toArray)).reduce(_+_)/models.length.toDouble).toArray.toList

    val meanProportionCols: List[Double] = (models.map(model =>
      DenseVector(model.proportionsCols.toArray)).reduce(_+_)/models.length.toDouble).toArray.toList

    val meanMu: DenseMatrix[DenseVector[Double]] = DenseMatrix.tabulate(L,K){(l, k)=>{
      models.map(m => DenseVector(m.gaussians(l)(k).mean.toArray)).reduce(_+_)/models.length.toDouble
    }}

    val meanCovariance: DenseMatrix[DenseMatrix[Double]] = DenseMatrix.tabulate(L,K){ (l, k)=>{
      models.map(m => DenseMatrix(m.gaussians(l)(k).cov.toArray).reshape(meanMu(0,0).length,meanMu(0,0).length)).reduce(_+_)/models.length.toDouble
    }}

    val meanGaussians: List[List[MultivariateGaussian]] = (0 until L).map(l => {
      (0 until K).map(k => {
        new MultivariateGaussian(
          Vectors.dense(meanMu(l,k).toArray),
          denseMatrixToMatrix(meanCovariance(l,k)))
      }).toList
    }).toList

    LatentBlockModel(meanProportionRows, meanProportionCols, meanGaussians)
  }


  def run(data: RDD[(Int, Array[DenseVector[Double]])],
          EMMethod: String= "SEMGibbs",
          verbose: Boolean = false,
          nConcurrent:Int=5,
          nTryMaxPerConcurrent:Int=10,
          initMethod: String = "randomPartition")(implicit ss: SparkSession): Map[String,Product] = {

    val allRes = (0 until nConcurrent).map(nTry => {
      if(verbose) {println()
        println("> n° launch "+(1+nTry).toString+"/"+nConcurrent.toString)}
      this.initAndRunTry(data,EMMethod, nTryMax = nTryMaxPerConcurrent, verbose=verbose)
    }).toList
    val allLikelihoods: DenseVector[Double] = DenseVector(allRes.map(_("LogLikelihood").asInstanceOf[List[Double]].last).toArray)

    allRes(argmax(allLikelihoods))
  }

  def initAndRunTry(data: RDD[(Int, Array[DenseVector[Double]])],
                    EMMethod:String = "SEMGibbs",
                    nTry: Int = 1,
                    nTryMax: Int = 50,
                    verbose: Boolean= true)(implicit ss: SparkSession): Map[String,Product] = {

    var t0 = System.nanoTime()
    val p:Int = data.take(1).head._2.length
    val n:Int = data.count().toInt
    var logLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity

    if(nTry > nTryMax){
      return Map("Model" -> new LatentBlockModel().asInstanceOf[Product],
        "RowPartition" -> List.fill(n)(0),
        "ColPartition" -> List.fill(p)(0),
        "LogLikelihood" -> logLikelihoodList.toList.asInstanceOf[Product])
    }

    Try(this.initAndLaunch(data, EMMethod,n,p,verbose=verbose)) match {
      case Success(v) =>
        if(verbose) {println()
        printTime(t0, EMMethod+" LBM Spark")}
        Success(v).get.asInstanceOf[Map[String, Product]]
      case Failure(e) =>
        if(verbose) {if(nTry==1){
          print("Algorithm "+ EMMethod+" didn't converge to an appropriate solution, trying again..\n" +
            "n° try: "+nTry.toString+"")
        } else {print(", "+nTry.toString)}}
        this.initAndRunTry(data, EMMethod,  nTry+1, nTryMax=nTryMax, verbose=verbose)
    }
  }

  def initAndLaunch(data: RDD[(Int, Array[DenseVector[Double]])],
                    EMMethod: String,
                    n: Int,
                    p: Int,
                    verbose:Boolean=true
                    )(implicit ss: SparkSession): Map[String,Product]= {

    val (initialModel, initialColPartition): (LatentBlockModel, List[Int]) = providedInitialModel match {
      case Some(model) =>
        (model, (0 until p).map(_ => sample(model.proportionsCols)).toList)
      case None => Initialization.initFromRandomPartition(data,K,L,n,p, this.fullCovarianceHypothesis)
    }
    launchEM(data, EMMethod, initialColPartition,initialModel,n, p,verbose)
  }

  def launchEM(data: RDD[(Int, Array[DenseVector[Double]])],
               EMMethod: String,
               initialColPartition: List[Int],
               initialModel: LatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean=false)(implicit ss: SparkSession): Map[String,Product] = {

    require(List("SEMGibbs","VariationalEM").contains(EMMethod),
      "EM Method provided "+EMMethod+" is not available. Possible options: SEMGibbs, VariationalEM")
    EMMethod match {
      case "SEMGibbs" => SEMGibbs(data, initialColPartition, initialModel,n,p, verbose)
      case "VariationalEM" => variationalEM(data, initialColPartition, initialModel,n,p, verbose)
    }
  }

  def SEMGibbs(data: RDD[(Int, Array[DenseVector[Double]])],
               initialColPartition: List[Int],
               initialModel: LatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean=true)(implicit ss: SparkSession): Map[String, Product] = {

    var precColPartition = initialColPartition
    var precModel = initialModel
    var precData: RDD[(Int, Array[DenseVector[Double]], Int)] = data.map(r => (r._1, r._2, 0))
    precData = precModel.SEMGibbsExpectationStep(precData, precColPartition, verbose = verbose)._1
    var completeLogLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]():+
      Double.NegativeInfinity :+
      precModel.completelogLikelihood(precData,precColPartition)

    var cachedModels = List(precModel)
    if(verbose){println(">>> Initial model")}
    if(verbose){println("Loglikelihood: "+completeLogLikelihoodList.last.toString)}

    var hasConverged: Boolean = false
    var t0 = System.nanoTime
    var iter =0
    do {
      iter += 1
      if(verbose){println(">>> iter: "+iter.toString)}

      val (newData, newColPartition) = precModel.SEMGibbsExpectationStep(precData, precColPartition, verbose = verbose)
      if(verbose){t0 = printTime(t0, "SE")}

      val newModel = precModel.SEMGibbsMaximizationStep(newData, newColPartition, this.fullCovarianceHypothesis, verbose)
      if(verbose){t0 = printTime(t0, "Maximization")}

      precModel = newModel
      precData = newData
      precColPartition =  newColPartition
      cachedModels = cachedModels :+ precModel
      completeLogLikelihoodList += precModel.completelogLikelihood(precData,precColPartition)
      if(verbose){println("Loglikelihood: "+completeLogLikelihoodList.last.toString)}

      hasConverged = abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(1).last)<precision &
        abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(2).last)<precision

    } while (iter < (maxBurninIterations+maxIterations) & !hasConverged)


    val iclList = completeLogLikelihoodList.map(ll => precModel.ICL(ll,n,p,true)).toList.drop(1)

    val resModel = if (!hasConverged){
      computeMeanModels(cachedModels.drop(1+maxBurninIterations))
    } else cachedModels.last


    Map("Model" -> resModel.asInstanceOf[Product],
      "RowPartition" -> precData.map(row => (row._1, row._3)).collect().sortBy(_._1).map(_._2).toList.asInstanceOf[Product],
      "ColPartition" -> precColPartition.asInstanceOf[Product],
      "LogLikelihood" -> completeLogLikelihoodList.toList.drop(1).asInstanceOf[Product],
      "ICL" -> iclList.drop(1).asInstanceOf[Product])

  }

  def variationalEM(data: RDD[(Int, Array[DenseVector[Double]])],
                    initialColPartition: List[Int],
                    initialModel: LatentBlockModel,
                    n:Int,
                    p:Int,
                    verbose:Boolean=true,
                    withCachedModels: Boolean= true)(implicit ss: SparkSession): Map[String,Product] =  {

    var precData: RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])] = data.map(r => (r._1, r._2, DenseVector(0D)))
    var precModel = initialModel
    var precJointLogDistribCols: DenseMatrix[Double] = DenseMatrix(partitionToBelongingProbabilities(initialColPartition):_*)

    var completeLogLikelihoodList: ListBuffer[Double] = ListBuffer.empty ++= List(
      Double.NegativeInfinity,
      precModel.completeLogLikelihoodFromMAPClusters(precData,precJointLogDistribCols))

    var hasConverged: Boolean = false
    var t0 = System.nanoTime
    var iter =0
    do {
      iter += 1
      val (newData, newJointLogDistribCols) = precModel.expectationStep(precData, precJointLogDistribCols, verbose = verbose)
      precModel = precModel.maximizationStep(newData, newJointLogDistribCols, this.fullCovarianceHypothesis, verbose)
      precJointLogDistribCols = newJointLogDistribCols
      precData = newData
      completeLogLikelihoodList += precModel.completeLogLikelihoodFromMAPClusters(precData,precJointLogDistribCols)

      hasConverged = abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(1).last)<precision &
        abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(2).last)<precision
      hasConverged = abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(1).last)<precision &
        abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(2).last)<precision
    } while (iter < (maxBurninIterations+maxIterations) & !hasConverged)

    val iclList = completeLogLikelihoodList.map(ll => precModel.ICL(ll, n, p, fullCovariance = true))

    Map("Model" -> precModel.asInstanceOf[Product],
      "RowPartition" -> rowPartitionFromRDDPartition(precData).asInstanceOf[Product],
      "ColPartition" -> precJointLogDistribCols(*, ::).map( dv => argmax(dv)).toArray.toList.asInstanceOf[Product],
      "LogLikelihood" -> completeLogLikelihoodList.toList.asInstanceOf[Product],
      "ICL" -> iclList.toList.asInstanceOf[Product])
  }

}
