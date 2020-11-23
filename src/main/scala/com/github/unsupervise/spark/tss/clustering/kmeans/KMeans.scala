/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */


package com.github.unsupervise.spark.tss.clustering.kmeans

import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools.printTime

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD

import scala.util.{Failure, Random, Success, Try}


class KMeans private(private var K: Int,
                     private var maxIterations: Int,
                     private var seed: Long) extends Serializable {

  val precision = 1e-5

  /**
    * Constructs a default instance. The default parameters are {k: 2, convergenceTol: 0.01,
    * maxIterations: 100, seed: random}.
    */
  def this() = this(2, 10, seed = Random.nextLong())

  /**
    * Set the number of Gaussians in the mixture model.  Default: 2
    */
  def setK(K: Int): this.type = {
    require(K > 0,
      s"Every numbers of row clusters must be positive but got $K")
    this.K = K
    this
  }

  /**
    * Return the number of row cluster number in the latent block model
    */
  def getK: Int = K

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
    * Return the maximum number of iterations allowed
    */
  def getMaxIterations: Int = maxIterations

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

  def run(data: RDD[(Int, Array[DenseVector[Double]])],
                    nTry: Int = 1,
                    nTryMax: Int = 10,
                    initMethod: String = "KMeansPP",
                    verbose: Boolean=false): Map[String,Product] = {

    var t0 = System.nanoTime()
    val n:Int = data.take(1).head._2.length

    if(nTry > nTryMax){return Map("Model" -> new KMeansModel(), "Partition" -> List.fill(n)(0))}

    Try(this.initAndLaunch(data, initMethod = initMethod, verbose=verbose)) match {
      case Success(v) =>
        if(verbose){println()
        printTime(t0, "KMeans Spark")}
        Success(v).get.asInstanceOf[Map[String, Product with Serializable]]
      case Failure(e) =>
        if(verbose){
          if(nTry==1){
          print("Algorithm KMeans didn't converge to an appropriate solution, trying again..\n" +
            "n° try: "+nTry.toString+"")
        } else {print(", "+nTry.toString)}}
        this.run(data, nTry+1,nTryMax, initMethod=initMethod, verbose=verbose)
    }
  }


  def initAndLaunch(data: RDD[(Int, Array[DenseVector[Double]])],
                    verbose:Boolean=true,
                    initMethod: String = ""): Map[String,Product]= {
    val n:Int = data.count().toInt
    val initialModel = Initialization.initialize(data,this,n,verbose,initMethod)
    kmeans(data, initialModel, verbose)
  }

  def kmeans(data: RDD[(Int, Array[DenseVector[Double]])],
             initialModel: KMeansModel,
             verbose:Boolean=true): Map[String,Product] = {

    var precData: RDD[(Int, Array[DenseVector[Double]], Int)] = data.map(r => (r._1, r._2, 0))
    var precModel = initialModel
    var iter = 0

    if(verbose){println(">>>>> Initial model")}

    do {
      iter +=1
      if(verbose){println(">>>>> iter: "+iter.toString)}

      val newData = precModel.ClassificationExpectationStep(precData)
      val newModel = precModel.MaximizationStep(newData)
      precModel = newModel
      precData = newData

    } while (iter < maxIterations)

    val partition = precData.map(row => (row._1, row._3)).collect().sortBy(_._1).map(_._2).toList

    Map("Model" -> precModel,
        "Partition" -> partition)
  }

}
