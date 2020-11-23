/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.lbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ToolsRDD._
import Tools._
import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, sum}
import breeze.numerics.{exp, log}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.{MultivariateGaussian, _}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

case class LatentBlockModel(proportionsRows: List[Double],
                            proportionsCols: List[Double],
                            gaussians: List[List[MultivariateGaussian]]) {
  val precision = 1e-5

  // Auxiliary constructor also takes a String?? (compile error)
  def this() {
    this(List(0D),List(0D),List(List(new MultivariateGaussian(
      Vectors.dense(Array(0D)),
      denseMatrixToMatrix(DenseMatrix(1D))))))
  }

  def this(K:Int,
           L:Int) {
    this(
      List.fill(K)(1/K.toDouble),
      List.fill(L)(1/L.toDouble),
      List.fill(L)(List.fill(K){new MultivariateGaussian(
        Vectors.dense(Array(0D)),
        denseMatrixToMatrix(DenseMatrix(1D)))}))
  }

  // Version to call inside the SEM algorithm
  def SEMGibbsExpectationStep(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                              colPartition: List[Int],
                              nIter: Int = 3,
                              verbose: Boolean = true)(implicit ss: SparkSession): (RDD[(Int, Array[DenseVector[Double]], Int)], List[Int]) = {

    var newData: RDD[(Int, Array[DenseVector[Double]], Int)] = drawRowPartition(data, colPartition)
    var newColPartition: List[Int] = drawColPartition(newData)
    var k: Int = 0
    while (k < nIter) {
      newData = drawRowPartition(newData, newColPartition)
      newColPartition = drawColPartition(newData)
      k += 1
    }
    (newData, newColPartition)
  }

  def drawRowPartition(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                       colPartition: List[Int])(implicit ss: SparkSession): RDD[(Int, Array[DenseVector[Double]], Int)] = {
    val jointLogDistribRows: RDD[(Int, Array[DenseVector[Double]], List[Double])] = computeJointLogDistribRowsFromSample(data, colPartition)
    jointLogDistribRows.map(x => {
      (x._1,
        x._2, {
        val LSE = logSumExp(x._3)
        sample(x._3.map(e => exp(e - LSE)))
      })
    })
  }

  def computeJointLogDistribRowsFromSample(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                                           colPartition: List[Int])(implicit ss: SparkSession):
  RDD[(Int, Array[DenseVector[Double]], List[Double])] = {

    val logPiBc = ss.sparkContext.broadcast(DenseVector(this.proportionsRows.map(log(_)).toArray))
    val colPartBc = ss.sparkContext.broadcast(colPartition)
    val gaussianBc = ss.sparkContext.broadcast(this.gaussians)
    data.map(row => {
      (row._1,
        row._2,
        //Row per row-cluster log likelihood is obtained by taking each row-cluster log likelihood of the row
        //by taking for each feature the gaussian corresponding to the the feature block, together with the row cluster considered
        //For example for feature 1, the row assignment to each of the K clusters is obtained by first retrieving the col cluster c1 of feature 1
        //then computing the logpdf of the row's feature 1 data using the gaussian(c1, k). 
        //The overall row membership to cluster k is done by summing up all log probas of its features to k
        (0 until this.K).map(k => {
          row._2.indices.map(j => {
            gaussianBc.value(colPartBc.value(j))(k).logpdf(Vectors.dense(row._2(j).toArray))
          }).sum + logPiBc.value(k)
        }).toList)
    })
  }

  def drawColPartition(data: RDD[(Int, Array[DenseVector[Double]], Int)])(implicit ss: SparkSession): List[Int] = {

    val jointLogDistribCols: List[List[Double]] = computeJointLogDistribColsFromSample(data)
    jointLogDistribCols.map(x => {
      val LSE = logSumExp(x)
      sample(x.map(e => exp(e - LSE)))
    })
  }

  def computeJointLogDistribColsFromSample(data: RDD[(Int, Array[DenseVector[Double]], Int)])
                                          (implicit ss: SparkSession): List[List[Double]] = {

    val logPiCols: DenseVector[Double] = DenseVector(this.proportionsCols.map(log(_)).toArray)
    val gaussianBc = ss.sparkContext.broadcast(this.gaussians)

    val D: RDD[DenseMatrix[Double]] =
      data.map(row => {
        row._2.indices.map(j => {
          DenseMatrix((0 until this.L).map(l => {
            this.gaussians(l)(row._3).logpdf(Vectors.dense(row._2(j).toArray))
          }).toArray)
        }).reduce((a, b) => DenseMatrix.vertcat(a, b))
      })

    val prob = D.reduce(_ + _)
    val sumProb = prob(*, ::).map(dv => dv.toArray.toList).toArray.toList.zipWithIndex.map(e =>
      (DenseVector(e._1.toArray) + logPiCols).toArray.toList)
    sumProb
  }

  // Version to call outside (integrate row and col partition initialisation)
  def StochasticExpectationStepAP(data: RDD[(Int, Array[DenseVector[Double]])],
                                  p: Int,
                                  nIter: Int = 5,
                                  verbose: Boolean = true)(implicit ss: SparkSession):
  RDD[(Int, Array[DenseVector[Double]], List[(Int,Int)])] = {

    val dataWithDummyRowPartition: RDD[(Int, Array[DenseVector[Double]], Int)] = data.map(r => (r._1, r._2, 0))
    val colPartition: List[Int] = (0 until p).map(_ => sample(this.proportionsCols)).toList

    var newData: RDD[(Int, Array[DenseVector[Double]], Int)] = drawRowPartition(dataWithDummyRowPartition, colPartition)

    var newColPartition: List[Int] = drawColPartition(newData)
    var k: Int = 0
    while (k < nIter) {
      newData = drawRowPartition(newData, newColPartition)
      newColPartition = drawColPartition(newData)
      k += 1
    }
    newData.map(row => (row._1, row._2, List.fill(row._2.length)(row._3) zip newColPartition))
  }

  def SEMGibbsMaximizationStep(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                               colPartition: List[Int],
                               fullCovariance: Boolean,
                               verbose: Boolean = true)(implicit ss: SparkSession): LatentBlockModel = {

    println("maximization step")
    val partitionPerColBc = ss.sparkContext.broadcast(DenseVector(colPartition:_*))
    val (means, covMat, sizeBlock, _) = getMeansAndCovariances(data, partitionPerColBc,L,K,fullCovariance)

    val Models: List[List[MultivariateGaussian]] =
      (0 until L).map(l => {
        (0 until K).map(k => {
          new MultivariateGaussian(
            Vectors.dense(means(l)(k).toArray),
            denseMatrixToMatrix(covMat(l)(k)))
        }).toList
      }).toList

    val proportions = proportionFromLBMSizeBlock(sizeBlock)

    LatentBlockModel(proportions._1, proportions._2, Models)
  }

  def expectationStep(data: RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])],
                      jointLogDistribCols: DenseMatrix[Double],
                      nIter: Int = 5,
                      verbose: Boolean = true)(implicit ss: SparkSession): (RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])], DenseMatrix[Double]) = {

    var dataWithNewRowJointLogDistrib: RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])] = computeJointDistribRows(data, jointLogDistribCols)
    var newColJointLogDistrib: DenseMatrix[Double] = computeProbBelongingtoCol(dataWithNewRowJointLogDistrib)
    if (nIter > 1) {
      for (_ <- 1 until nIter) {
        dataWithNewRowJointLogDistrib = computeJointDistribRows(dataWithNewRowJointLogDistrib, jointLogDistribCols)
        newColJointLogDistrib = computeProbBelongingtoCol(dataWithNewRowJointLogDistrib)
      }
    }
    (dataWithNewRowJointLogDistrib, newColJointLogDistrib)
  }

  def computeJointDistribRows(dataRDD: RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])],
                              jointLogDistribCol: DenseMatrix[Double])(implicit ss: SparkSession): RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])] = {

    val p = jointLogDistribCol.rows
    val jointLogDistribColBc = ss.sparkContext.broadcast(jointLogDistribCol)
    val logPi: List[Double] = this.proportionsRows.map(log(_))
    val logPdf: RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])] =
      dataRDD.map(row => {
        (row._1,
          row._2, {
          val prob = DenseVector((0 until this.K).map(k => {
            val listWLogf: DenseVector[Double] = DenseVector(((0 until p) cross (0 until L)).map(jl => {
              jointLogDistribColBc.value(jl._1, jl._2)*
                this.gaussians(jl._2)(k).logpdf(Vectors.dense(row._2(jl._1).toArray))
            }).toArray)
            logPi(k) + sum(listWLogf)
          }).toArray)
          prob
        })
      })

    val res = logPdf.map(row => {
      (row._1, row._2, {
        val LSE = logSumExp(row._3)
        row._3.map(e => exp(e - LSE))})
    })
    res
  }

  def K: Int = proportionsRows.length

  def L: Int = proportionsCols.length

  def computeProbBelongingtoCol(dataRDD: RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])]): DenseMatrix[Double] = {

    val logRho: DenseVector[Double] = DenseVector(this.proportionsCols.map(log(_)).toArray)

    val logPdfPerRow: RDD[DenseMatrix[Double]] = dataRDD.map(row => {
      (0 until this.L).map(l => {
        val sumWLogf: DenseMatrix[Double] = DenseMatrix(row._2.map(V => {
          (0 until this.K).map(k => {
            row._3(k) * this.gaussians(l)(k).logpdf(Vectors.dense(V.toArray))
          }).sum
        })).t
        sumWLogf

      }).reduce((a, b) => DenseMatrix.horzcat(a, b))
    })

    val sumLogPdf = logPdfPerRow.reduce(_ + _)
    val sumJointPdf = sumLogPdf(*, ::).map(vec => vec + logRho)
    val probBelonging = sumJointPdf(*, ::).map(row => {
      val LSE = logSumExp(row)
      row.map(e => exp(e - LSE))
    })

    probBelonging
  }

  def ICL(completelogLikelihood: Double,
          n: Double,
          p: Double,
          fullCovariance: Boolean): Double = {

    val dimVar = this.gaussians.head.head.mean.size
    val nParamPerComponent = if(fullCovariance){
      dimVar+ dimVar*(dimVar+1)/2D
    } else {
      2D * dimVar
    }
    completelogLikelihood - log(n)*(this.K - 1)/2D - log(p)*(L-1)/2D - log(n*p)*(K*L*nParamPerComponent)/2D
  }

  def completeLogLikelihoodFromMAPClusters(data: RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])],
                                           jointLogDistribCols: DenseMatrix[Double]): Double = {

    val dataWithMAPRowPartition = data.map(row => (row._1, row._2, argmax(row._3)))
    val colPartitionFromMAP = jointLogDistribCols(*, ::).map(argmax(_)).toArray.toList
    completelogLikelihood(dataWithMAPRowPartition,colPartitionFromMAP)
  }

  def completelogLikelihood(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                            colPartition: List[Int]): Double = {

    val logRho: List[Double] = this.proportionsCols.map(log(_))
    val logPi: List[Double]  = this.proportionsRows.map(log(_))

    data.map(row => {
      row._2.indices.map(j => {
        val l = colPartition(j)
        logPi(row._3)
        + logRho(l)
        + this.gaussians(colPartition(j))(row._3).logpdf(Vectors.dense(row._2(j).toArray))
      }).sum
    }).sum
  }

  def maximizationStep(data: RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])],
                       jointLogDistribCols: DenseMatrix[Double],
                       fullCovariance: Boolean = true,
                       verbose: Boolean = true): LatentBlockModel = {

    val newColProportions: List[Double] = (sum(jointLogDistribCols(::, *)).t / jointLogDistribCols.rows.toDouble).toArray.toList
    require(newColProportions.min>this.precision, "Algorithm could not converge: empty block")
    val newRowProportions: List[Double] = {
      val sumProportionsAndn = data.map(r => (r._3, 1)).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
      sumProportionsAndn._1 / sumProportionsAndn._2.toDouble
    }.toArray.toList
    require(newRowProportions.min>this.precision, "Algorithm could not converge: empty block")

    val sumValuesAndSumCoefsPerRow = data.map(row => {
      val rowMode: DenseMatrix[(DenseVector[Double], Double)] = DenseMatrix.tabulate(this.K, this.L){(k, l) => {
        row._2.indices.map(j => {
          (row._3(k) * jointLogDistribCols(j, l) * row._2(j),
            row._3(k) * jointLogDistribCols(j, l)): (DenseVector[Double], Double)
        }).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
      }
      }
      (rowMode.map(_._1), rowMode.map(_._2))
    })

    val sumValues: DenseMatrix[DenseVector[Double]] = sumValuesAndSumCoefsPerRow.map(_._1).reduce(_ + _)
    val sumCoefs: DenseMatrix[Double] = sumValuesAndSumCoefsPerRow.map(_._2).reduce(_ + _)

    val modes = DenseMatrix.tabulate[DenseVector[Double]](this.K, this.L) { (k, l) =>
    {sumValues(k, l) / sumCoefs(k, l)}}

    val unitCovFunction = unitCovFunc(fullCovariance)
    val sumProductCentered: DenseMatrix[DenseMatrix[Double]] = data.map(row => {
      DenseMatrix.tabulate(this.K, this.L) { (k, l) => {
        row._2.indices.map(j => {
          val centeredRow = row._2(j) - modes(k, l)
          row._3(k) * jointLogDistribCols(j, l) * unitCovFunction(centeredRow)
        }).reduce(_ + _)
      }}
    }).reduce(_ + _)

    val covariances = DenseMatrix.tabulate[DenseMatrix[Double]](this.K, this.L) { (k, l) =>
    {sumProductCentered(k, l) / sumCoefs(k, l)}}

    val Models: List[List[MultivariateGaussian]] =
      (0 until this.L).map(l => {
        (0 until this.K).map(k => {
          new MultivariateGaussian(
            Vectors.dense(modes(k, l).toArray),
            denseMatrixToMatrix(covariances(k, l)))
        }).toList
      }).toList

    LatentBlockModel(newRowProportions, newColProportions, Models)

  }
}

