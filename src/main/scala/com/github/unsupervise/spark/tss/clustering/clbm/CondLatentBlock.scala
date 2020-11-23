/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.clbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import breeze.linalg.{DenseMatrix, DenseVector, argmax, max}
import breeze.numerics.abs
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer
import scala.util.{Failure, Random, Success, Try}


class CondLatentBlock (private var KVec: List[Int] = List(2,2),
                      private var maxIterations: Int = 10,
                      private var maxBurninIterations: Int = 10,
                      private var fullCovarianceHypothesis: Boolean = true,
                      private var seed: Long = Random.nextLong()) extends Serializable {

  val precision = 1e-2

  // an initializing model can be provided rather than using the
  // default random starting point
  private var providedInitialModel: Option[CondLatentBlockModel] = None
  private var providedInitialColPartition: Option[List[Int]] = None

  /**
    * Set the initial GMM starting point, bypassing the random initialization.
    * You must call setK() prior to calling this method, and the condition
    * (model.k == this.k) must be met; failure will result in an IllegalArgumentException
    */
  def setInitialModel(model: CondLatentBlockModel): this.type = {
    require(model.KVec == KVec,
      s"Mismatched row cluster number (model.KVec ${model.KVec} != KVec $KVec)")
    providedInitialModel = Some(model)
    this
  }

  def setInitialColPartition(colPartition: List[Int]): this.type = {
    val uniqueCol = colPartition.distinct
    require(uniqueCol.length == KVec.length,
      s"Mismatched column cluster number (colPartition.distinct.length ${uniqueCol.length} != KVec.length ${KVec.length})")
    providedInitialColPartition = Some(colPartition)
    this
  }


  /**
    * Return the user supplied initial GMM, if supplied
    */
  def getInitialModel: Option[CondLatentBlockModel] = providedInitialModel

  /**
    * Set the number of Gaussians in the mixture model.  Default: 2
    */
  def setKVec(KVec: List[Int]): this.type = {
    require(KVec.forall(_ > 0),
      s"Every numbers of row clusters must be positive but got $KVec")
    this.KVec = KVec
    this
  }


  /**
    * Return the number of row cluster number in the latent block model
    */
  def getKVec: List[Int] = KVec


  /**
    * Return the number of column cluster number in the latent block model
    */
  def getL: Int = KVec.length

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

  def meanModels(models: List[CondLatentBlockModel]): CondLatentBlockModel = {

    val nTotalIter = models.length.toDouble
    val meanProportionRows: List[List[Double]] = models.head.proportionsRows.indices.map(idx => {
      models.map(model =>
        DenseVector(model.proportionsRows(idx).toArray) / nTotalIter).reduce(_ + _).toArray.toList
    }).toList

    val meanProportionCols: List[Double] = models.map(model =>
      DenseVector(model.proportionsCols.toArray) / nTotalIter).reduce(_ + _).toArray.toList

    val meanMu: List[List[DenseVector[Double]]] =
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k_l => {
          DenseVector(models.map(model => DenseVector(model.gaussians(l)(k_l).mean.toArray)).reduce(_ + _).toArray) / nTotalIter
        }).toList
      }).toList

    val d:Int = meanMu.head.head.length
    val meanCovariance: List[List[DenseMatrix[Double]]] =
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k_l => {
          models.map(m => DenseMatrix(m.gaussians(l)(k_l).cov.toArray).reshape(d,d)).reduce(_ + _) / nTotalIter
        }).toList
      }).toList

    val meanGaussians = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        new MultivariateGaussian(Vectors.dense(meanMu(l)(k_l).toArray), denseMatrixToMatrix(meanCovariance(l)(k_l)))
      }).toList
    }).toList

    CondLatentBlockModel(meanProportionRows, meanProportionCols, meanGaussians)
  }

  def computeMeanModels(models: List[CondLatentBlockModel]): CondLatentBlockModel = {

    val nTotalIter = models.length.toDouble
    val meanProportionRows: List[List[Double]] = models.head.proportionsRows.indices.map(idx => {
      models.map(model =>
        DenseVector(model.proportionsRows(idx).toArray) / nTotalIter).reduce(_ + _).toArray.toList
    }).toList

    val meanProportionCols: List[Double] = models.map(model =>
      DenseVector(model.proportionsCols.toArray) / nTotalIter).reduce(_ + _).toArray.toList

    val meanMu: List[List[DenseVector[Double]]] =
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k_l => {
          DenseVector(models.map(model => DenseVector(model.gaussians(l)(k_l).mean.toArray)).reduce(_ + _).toArray) / nTotalIter
        }).toList
      }).toList

    val meanGaussians: List[List[MultivariateGaussian]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        val covMat = models.map(m => matrixToDenseMatrix(m.gaussians(l)(k_l).cov)).reduce(_ + _) / nTotalIter
        new MultivariateGaussian(Vectors.dense(meanMu(l)(k_l).toArray), denseMatrixToMatrix(covMat))
      }).toList
    }).toList

    CondLatentBlockModel(meanProportionRows, meanProportionCols, meanGaussians)
  }

  def initAndRunTry(data: RDD[(Int, Array[DenseVector[Double]])],
                    EMMethod:String = "SEMGibbs",
                    nTry: Int = 1,
                    nTryMax: Int = 50,
                    initMethod: String = "randomPartition",
                    verbose: Boolean=false)(implicit ss: SparkSession): Map[String,Product] = {

    val t0 = System.nanoTime()
    val logLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity
    val ICL: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity
    val p:Int = data.take(1).head._2.length

    if(nTry > nTryMax){
      return Map("Model" -> new CondLatentBlockModel(),
        "ColPartition" -> List.fill(p)(0),
        "LogLikelihood" -> logLikelihoodList.toList,
        "ICL" -> ICL.toList)
    }

    Try(this.initAndLaunch(data, EMMethod, initMethod = initMethod, verbose=verbose)) match {
      case Success(v) =>
        if(verbose){println()
        printTime(t0, EMMethod+" CLBM Spark")}
        Success(v).get.asInstanceOf[Map[String, Product with Serializable]]
      case Failure(_) =>
        if(verbose){
          if(nTry==1){
          print("Algorithm "+ EMMethod+" didn't converge to an appropriate solution, trying again..\n" +
            "n° try: "+nTry.toString+"")
        } else {print(", "+nTry.toString)}}
        this.initAndRunTry(data, EMMethod, nTry+1,nTryMax, initMethod=initMethod, verbose=verbose)
    }
  }

  def initAndLaunch(data: RDD[(Int, Array[DenseVector[Double]])],
                    EMMethod: String,
                    verbose:Boolean=true,
                    initMethod: String = "randomPartition")(implicit ss: SparkSession): Map[String,Product]= {

    val p:Int = data.take(1).head._2.length
    val n:Int = data.count().toInt

    val (initialModel, initialColPartition) = providedInitialModel match {
      case Some(model) =>
        require(initMethod.isEmpty,
          s"An initial model has been provided but initMethod argument has also been set to $initMethod. " +
            s"Please make a choice: do not set an initMethod or do not provide an initial model.")
        (model,
          providedInitialColPartition match {
            case Some(colPartition) => colPartition
            case None => (0 until p).map(_ => sample(model.proportionsCols)).toList
          })
      case None => Initialization.initialize(data,this,EMMethod,n,p,verbose,initMethod)
    }

    launchEM(data, EMMethod, initialColPartition,initialModel,n, p,verbose)
  }

  def launchEM(data: RDD[(Int, Array[DenseVector[Double]])],
               EMMethod: String,
               initialColPartition: List[Int],
               initialModel: CondLatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean=false)(implicit ss: SparkSession): Map[String,Product] = {

    require(List("SEMGibbs").contains(EMMethod),
      "EM Method provided "+EMMethod+" is not available. Possible options: SEMGibbs, VariationalEM")
    EMMethod match {
      case "SEMGibbs" => SEMGibbs(data, initialColPartition, initialModel,n,p, verbose)
    }
  }

  def run(data: RDD[(Int, Array[DenseVector[Double]])],
          EMMethod: String= "SEMGibbs",
          verbose: Boolean = false,
          nConcurrent:Int=10,
          nTryMaxPerConcurrent:Int=20,
          initMethod: String = "randomPartition")(implicit ss: SparkSession): Map[String,Product] = {

    val allRes = (0 until nConcurrent).map(nTry => {
      if(verbose){println("> n° launch "+(1+nTry).toString+"/"+nConcurrent.toString)}
      this.initAndRunTry(data,EMMethod, nTryMax = nTryMaxPerConcurrent, initMethod = initMethod, verbose=verbose)
    }).toList
    val allLikelihoods: DenseVector[Double] = DenseVector(allRes.map(_("LogLikelihood").asInstanceOf[List[Double]].last).toArray)
    if(verbose){println("best LogLikelihood: " + max(allLikelihoods).toString)}
    allRes(argmax(allLikelihoods))
  }

  def SEMGibbs(data: RDD[(Int, Array[DenseVector[Double]])],
               initialColPartition: List[Int],
               initialModel: CondLatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean=true,
               withCachedModels: Boolean= true)(implicit ss: SparkSession): Map[String,Product] = {

    require(this.fullCovarianceHypothesis, "in SEM-Gibbs mode, indep. covariance hyp. is not yet available," +
      " please set latentBlock.fullCovarianceHypothesis to true")
    var precData: RDD[(Int, Array[DenseVector[Double]], List[Int])] = data.map(r => (r._1, r._2, List(0,0)))
    var precColPartition = initialColPartition
    precData = initialModel.drawRowPartition(precData, precColPartition)
    var precModel = initialModel

    if(verbose){
      precData = precModel.SEMGibbsExpectationStep(precData, precColPartition, verbose = verbose)._1
    }

    var completeLogLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]():+ Double.NegativeInfinity :+ precModel.completelogLikelihood(precData,precColPartition)
    var cachedModels = List[CondLatentBlockModel]():+precModel

    if(verbose){println(">>> Initial model")}
    if(verbose){println("Loglikelihood: "+completeLogLikelihoodList.last.toString)}

    var t0 = System.nanoTime
    var iter = 0

    do {
      iter += 1
      if(verbose){println(">>> iter: "+iter.toString)}

      val (newData, newColPartition) = precModel.SEMGibbsExpectationStep(precData, precColPartition, verbose = verbose)

      if(verbose){t0 = printTime(t0, "SE")}

      precModel = precModel.SEMGibbsMaximizationStep(newData, newColPartition, n, verbose)

      if(verbose){t0 = printTime(t0, "Maximization")}

      precData = newData
      precColPartition = newColPartition

      cachedModels = cachedModels :+ precModel
      completeLogLikelihoodList += precModel.completelogLikelihood(precData,precColPartition)
      if(verbose){println("Loglikelihood: "+completeLogLikelihoodList.last.toString)}

    } while (iter < (maxBurninIterations+maxIterations) &
      !(abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(1).last)<precision &
        abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(2).last)<precision)  )

    val iclList = completeLogLikelihoodList.map(ll => precModel.ICL(ll,n,p,fullCovariance = true))



    val rowPartition: scala.List[scala.List[Int]] = Tools.getRowPartitionFromDataWithRowPartition(precData)

    var res = if (!(abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(1).last)<precision &
      abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(2).last)<precision)) {
      Map("Model" -> computeMeanModels(cachedModels.drop(1+maxBurninIterations)),
        "RowPartition" -> rowPartition,
        "ColPartition" -> precColPartition,
        "LogLikelihood" -> completeLogLikelihoodList.toList.drop(1),
        "ICL" -> iclList.toList.drop(1))
    } else {
      Map("Model" -> cachedModels.last,
        "RowPartition" -> rowPartition,
        "ColPartition" -> precColPartition,
        "LogLikelihood" -> completeLogLikelihoodList.toList.drop(1),
        "ICL" -> iclList.toList.drop(1))
    }

    if(withCachedModels) {res += ("CachedModels" -> cachedModels)}

    res
  }

}
