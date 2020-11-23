/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.funclbm

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


class FunCondLatentBlock(var KVec: List[Int] = List(1),
                         var maxIterations: Int = 10,
                         var maxBurninIterations: Int = 10,
                         var fullCovarianceHypothesis: Boolean = true,
                         var seed: Long = Random.nextLong(),
                         var updateLoadings: Boolean= false,
                         var updateLoadingStrategy: Double = 5,
                         var maxPcaAxis : Int = 5){

  // an initializing model can be provided rather than using the
  // default random starting point
  var precision = 1e-3
  private var providedInitialModel: Option[FunCondLatentBlockModel] = None
  private var providedInitialColPartition: Option[List[Int]] = None

  /**
    * Set the number of Gaussians in the mixture model.  Default: 2
    */
  def setKVec(KVec: List[Int]): this.type = {
    println("Inside override setKVec")
    require(KVec.forall(_ > 0),
      s"Every numbers of row clusters must be positive but got $KVec")
    this.KVec = KVec
    this
  }

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
    * Set the initial GMM starting point, bypassing the random initialization.
    * You must call setK() prior to calling this method, and the condition
    * (model.k == this.k) must be met; failure will result in an IllegalArgumentException
    */
  def setInitialModel(model: FunCondLatentBlockModel): this.type = {
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

  def computeMeanModels(models: List[FunCondLatentBlockModel], verbose: Boolean= false): FunCondLatentBlockModel = {

    val KVec = models.head.KVec

    val nTotalIter = models.length.toDouble
    if(verbose){println("Averaging " + nTotalIter.toInt.toString + " models")}
    val meanProportionRows: List[List[Double]] = models.head.proportionsRows.indices.map(idx => {
      models.map(model =>
        DenseVector(model.proportionsRows(idx).toArray) / nTotalIter).reduce(_ + _).toArray.toList
    }).toList

    val meanProportionCols: List[Double] = models.map(model =>
      DenseVector(model.proportionsCols.toArray) / nTotalIter).reduce(_ + _).toArray.toList

    val meanMu: List[List[DenseVector[Double]]] =
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k_l => {
          DenseVector(models.map(m => {
            DenseVector(m.gaussians(l)(k_l).mean.toArray)
          }).reduce(_ + _).toArray) / nTotalIter
        }).toList
      }).toList

    val meanCenterings: List[List[DenseVector[Double]]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        models.map(m => m.centerings(l)(k_l)).reduce(_ + _) / nTotalIter
      }).toList
    }).toList

    val meanLoadings: List[List[DenseMatrix[Double]]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        models.map(m => {
          m.loadings(l)(k_l)
        }).reduce(_ + _) / nTotalIter
      }).toList
    }).toList

    val meanGaussians: List[List[MultivariateGaussian]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        val covMat = models.map(m => matrixToDenseMatrix(m.gaussians(l)(k_l).cov)).reduce(_ + _) / nTotalIter
        new MultivariateGaussian(Vectors.dense(meanMu(l)(k_l).toArray), denseMatrixToMatrix(covMat))
      }).toList
    }).toList

    new FunCondLatentBlockModel(meanProportionRows, meanProportionCols, meanGaussians,  meanLoadings, meanCenterings)
  }

  def run(periodogram: RDD[(Int, Array[DenseVector[Double]])],
          EMMethod: String= "SEMGibbs",
          verbose: Boolean = false,
          nConcurrent:Int=10,
          nTryMaxPerConcurrent:Int=20,
          initMethod: String = "randomPartition")(implicit ss: SparkSession): Map[String,Product] = {
    val n:Int = periodogram.count().toInt
    val p:Int = periodogram.take(1).head._2.length

    val allRes = (0 until nConcurrent).map(nTry => {
      if(verbose){println("> n° launch "+(1+nTry).toString+"/"+nConcurrent.toString)}
      this.initAndRunTry(periodogram, n, p, nTryMax = nTryMaxPerConcurrent, initMethod = initMethod, verbose=verbose)
    }).toList
    val allLikelihoods: DenseVector[Double] = DenseVector(allRes.map(_("LogLikelihood").asInstanceOf[List[Double]].last).toArray)

    allRes(argmax(allLikelihoods))
  }


  def initAndRunTry(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                    n: Int,
                    p: Int,
                    EMMethod:String = "SEMGibbs",
                    nTry: Int = 1,
                    nTryMax: Int = 50,
                    initMethod: String = "randomPartition",
                    verbose: Boolean=false)(implicit ss: SparkSession): Map[String,Product] = {


    val t0 = System.nanoTime()
    val logLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity
    val ICL: ListBuffer[Double] = new ListBuffer[Double]() += Double.NegativeInfinity

    if(nTry > nTryMax){
      return Map("Model" -> new FunCondLatentBlockModel(),
        "RowPartition" -> List(List.fill(n)(0)),
        "ColPartition" -> List.fill(p)(0),
        "LogLikelihood" -> logLikelihoodList.toList,
        "ICL" -> ICL.toList)
    }

    Try(this.initAndLaunch(periodogram, n,p,EMMethod, initMethod = initMethod, verbose=verbose)) match {
      case Success(v) =>
        if(verbose){println()
          printTime(t0, EMMethod+" FunCLBM Spark")}
        Success(v).get.asInstanceOf[Map[String, Product with Serializable]]
      case Failure(_) =>
        if(verbose){
          if(nTry==1){
            print("Algorithm "+ EMMethod+" didn't converge to an appropriate solution, trying again..\n" +
              "n° try: "+nTry.toString+"")
          } else {print(", "+nTry.toString)}}
        this.initAndRunTry(periodogram, n, p,
          EMMethod,
          nTry+1,
          nTryMax,
          initMethod=initMethod,
          verbose=verbose)
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
        if(verbose){println("Initial model provided")}
        if(!initMethod.isEmpty){
          println(s"Warning: an initial model has been provided but initMethod argument has also been set to $initMethod. " +
            s"Continuing with provided initial model.")
        }
        (model, providedInitialColPartition match {
            case Some(colPartition) =>
              if(verbose){println("Initial column partition provided")}
              colPartition
            case None => (0 until p).map(_ => sample(model.proportionsCols)).toList
          })
      case None => Initialization.initialize(periodogram,this,n,p,
        this.fullCovarianceHypothesis,this.maxPcaAxis, verbose,initMethod)
    }

    require(List("SEMGibbs","VariationalEM").contains(EMMethod),
      "EM Method provided "+EMMethod+" is not available. Possible options: SEMGibbs")
    EMMethod match {
      case "SEMGibbs" => SEMGibbs(periodogram, initialColPartition, initialModel,n,p, verbose)
    }
  }


  def SEMGibbs(periodogram: RDD[(Int, Array[DenseVector[Double]])],
               initialColPartition: List[Int],
               initialModel: FunCondLatentBlockModel,
               n:Int,
               p:Int,
               verbose:Boolean)(implicit ss: SparkSession): Map[String,Product] = {

    if(verbose){println("SEMGibbs begins")}
    require(this.fullCovarianceHypothesis, "in SEM-Gibbs mode, indep. covariance hyp. is not available," +
      " please set latentBlock.fullCovarianceHypothesis to true")
    this.setUpdateLoadings(false)
    var precPeriodogram: RDD[(Int, Array[DenseVector[Double]], List[Int])] = periodogram.map(r => (r._1, r._2, List(0,0)))
    var precColPartition = initialColPartition
    precPeriodogram = initialModel.drawRowPartition(precPeriodogram, precColPartition)
    var precModel = initialModel

    if(verbose){
      precPeriodogram = precModel.SEMGibbsExpectationStep(precPeriodogram, precColPartition, 3, verbose = verbose)._1
    }

    var completeLogLikelihoodList: ListBuffer[Double] =
      new ListBuffer[Double]() :+
        Double.NegativeInfinity :+
        precModel.completelogLikelihood(
          projectInLowerSubspace(precPeriodogram,precColPartition, precModel.loadings, precModel.centerings),
          precColPartition)

    var cachedModels = List[FunCondLatentBlockModel]():+precModel

    if(verbose){println(">>> Initial model")}
    if(verbose){println("Loglikelihood: "+completeLogLikelihoodList.last.toString)}

    var t0 = System.nanoTime
    var iter =0
    var hasConverged: Boolean = false
    do {
      iter +=1
      if(verbose){println(">>> iter: "+iter.toString)}
      if(iter>this.updateLoadingStrategy){this.setUpdateLoadings(true)}
      if(verbose){println("Update loading: "+this.updateLoadings.toString)}
      val (newData, newColPartition) = precModel.SEMGibbsExpectationStep(precPeriodogram,
        precColPartition, 3, verbose = verbose)
      if(verbose){t0 = printTime(t0, "SE")}
      val newModel = precModel.SEMGibbsMaximizationStep(newData, newColPartition, fullCovarianceHypothesis, updateLoadings = updateLoadings, this.maxPcaAxis, verbose)
      if(verbose){t0 = printTime(t0, "Maximization")}

      precModel = newModel
      precPeriodogram = newData
      precColPartition = newColPartition
      completeLogLikelihoodList += precModel.completelogLikelihood(
        projectInLowerSubspace(precPeriodogram,precColPartition,precModel.loadings,precModel.centerings),
        precColPartition)

      cachedModels = cachedModels :+ precModel
      if(verbose){println("Loglikelihood: "+completeLogLikelihoodList.last.toString)}

      hasConverged = abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(1).last)<precision &
        abs(completeLogLikelihoodList.last - completeLogLikelihoodList.dropRight(2).last)<precision

    } while (iter < (maxBurninIterations+maxIterations) & !hasConverged)

    val iclList = completeLogLikelihoodList.map(ll => precModel.ICL(ll,n,p,fullCovariance = true))

    println("hasConverged: "+ hasConverged.toString)
    val resModel = if (!hasConverged){
      if(verbose){println("Algorithm did not converge, result is the average of after-burnin iterations")}
      computeMeanModels(cachedModels.drop(1+maxBurninIterations), verbose)
    } else {
      if(verbose){println("Algorithm converged !")}
      cachedModels.last
    }

    val rowMembershipPerRow: List[List[Int]] = precPeriodogram.map(row => (row._1, row._3)).collect().sortBy(_._1).map(_._2).toList
    val rowPartition: List[List[Int]] = precModel.KVec.indices.map(l =>
      rowMembershipPerRow.map(rowMembership => rowMembership(l))).toList

    Map("Model" -> resModel,
      "RowPartition" -> rowPartition,
      "ColPartition" -> precColPartition,
      "LogLikelihood" -> completeLogLikelihoodList.toList.drop(1),
      "ICL" -> iclList.toList.drop(1))

  }

}
