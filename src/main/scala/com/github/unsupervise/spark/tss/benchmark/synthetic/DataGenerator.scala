/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  * Adapted by: Anthony Coutant
  */

package com.github.unsupervise.spark.tss.benchmark.synthetic

import com.github.unsupervise.spark.tss.core.TSS
import com.github.unsupervise.spark.tss.core.Utils.sample

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import breeze.linalg.{DenseVector, DenseMatrix, diag}
import breeze.stats.distributions.MultivariateGaussian

import scala.util.Random

object DataGenerator {

    def random2DConditionalClusters(prototypes: List[List[List[Double]=> List[Double]]],
                                    sigma: Double,
                                    sizeClusterRow: List[List[Int]],
                                    sizeClusterCol: List[Int],
                                    mixingProportion: Double=0D,
                                    seriesLength: Int = 100)
                                    (implicit ss: SparkSession): TSS = {

        require(prototypes.map(_.length) == sizeClusterRow.map(_.length))
        require(sizeClusterCol.length == prototypes.length)
        require(sizeClusterRow.map(_.sum == sizeClusterRow.head.sum).forall(identity))

        import ss.implicits._

        //Time values, indexed from 1 to series length, for use with prototypes
        val indices:List[Double] = (1 to seriesLength).map(_/seriesLength.toDouble).toList
        //Generate data matrices given block prototypes and parameters
        val modes = prototypes.indices.map(l => {prototypes(l).indices.map(k => {DenseVector(prototypes(l)(k)(indices).toArray)}).toList}).toList
        //Generate random covariance matrices using the noise sigma parameter
        val covariances = prototypes.indices.map(l => {prototypes(l).indices.map(_ => {
        diag(DenseVector(Array.fill(indices.length){sigma}))}).toList}).toList
        //Generate a list of matrix for each cluster block
        val dataPerBlock = generateDataPerBlock(modes, covariances, sizeClusterRow, sizeClusterCol, mixingProportion)
        //Concat the list of matrices as a big matrix
        val data: DenseMatrix[DenseVector[Double]] = doubleReduce(dataPerBlock)
        //The matrix is returned as a RDD of [matrix x, matrix y, time values, series values]
        val mat = (0 until data.rows).map(i => {(0 until data.cols).map(j => {(i, j, indices, data(i, j).toArray.toList)}).toList}).reduce(_++_)
        //Transform it as a TSS
        val tss = TSS.fromRDD4(ss.sparkContext.parallelize(mat))
        //Create true cluster memberships dataframes
        //First for column (features) cluster
        val trueColPartition: List[Int] = sizeClusterCol.indices.map(idx => List.fill(sizeClusterCol(idx))(idx)).reduce(_ ++ _)
        //Do the same per row cluster inside each column cluster
        //Each series in row is included in one cluster per column cluster
        val trueRowPartition: List[List[Int]] = sizeClusterRow.indices.map(l => {
            sizeClusterRow(l).indices.map(k_l => { 
                List.fill(sizeClusterRow(l)(k_l)) {k_l} 
            }).reduce(_ ++ _)
        }).toList
        val trueColPartitionDF = ss.sparkContext.parallelize(trueColPartition.zipWithIndex.map(c => (c._2, c._1))).toDF("vName", "trueFeatureCluster")
        val trueRowPartitionDF = ss.sparkContext.parallelize(trueRowPartition.transpose.zipWithIndex.map(c => (c._2, c._1))).toDF("scid", "trueSeriesClusters")
        //Join true cluster values to TSS
        val tss2 = tss.addByLeftJoin(trueColPartitionDF, trueColPartitionDF("vName") === tss.getDecoratorColumn(TSS.VARNAME_DECORATORNAME)).drop("vName")
        val tss3 = tss2.addByLeftJoin(trueRowPartitionDF, trueRowPartitionDF("scid") === tss2.getDecoratorColumn(TSS.SIMULATIONID_DECORATORNAME)).drop("scid")
        tss3.addElementAt("trueSeriesCluster", "trueSeriesClusters", "trueFeatureCluster").drop("trueSeriesClusters")
    }

    def random2DClusters(prototypes: List[List[List[Double]=> List[Double]]],
                         sigma: Double,
                         sizeClusterRow: List[Int],
                         sizeClusterCol: List[Int],
                         mixingProportion: Double=0D,
                         seriesLength: Int = 100)
                         (implicit ss: SparkSession): TSS = {

        require(sizeClusterRow.length == prototypes.head.length)
        require(sizeClusterCol.length == prototypes.length)

        import ss.implicits._

        val K = prototypes.head.length
        val L = prototypes.length

        //Time values, indexed from 1 to series length, for use with prototypes
        val indices:List[Double] = (1 to seriesLength).map(_/seriesLength.toDouble).toList
        //Generate data matrices given block prototypes and parameters
        val modes = (0 until L).map(l => {(0 until K).map(k => {DenseVector(prototypes(l)(k)(indices).toArray)}).toList}).toList
        //Generate random covariance matrices using the noise sigma parameter
        val covariances = (0 until L).map(_ => {(0 until K).map(_ => {diag(DenseVector(Array.fill(indices.length){sigma}))}).toList}).toList
        //Generate a list of matrix for each cluster block
        val sizeClusterRowEachColumn = List.fill(L)(sizeClusterRow)
        val dataPerBlock = generateDataPerBlock(modes, covariances, sizeClusterRowEachColumn, sizeClusterCol, mixingProportion)
        //Concat the list of matrices as a big matrix
        val data: DenseMatrix[DenseVector[Double]] = doubleReduce(dataPerBlock)
        //The matrix is returned as a RDD of [matrix x, matrix y, time values, series values]
        val mat = (0 until data.rows).map(i => {(0 until data.cols).map(j => {(i, j, indices, data(i, j).toArray.toList)}).toList}).reduce(_++_)
        //Transform it as a TSS
        val tss = TSS.fromRDD4(ss.sparkContext.parallelize(mat))
        //Create true cluster memberships dataframes
        //First for column (features) cluster
        val trueColPartition: List[Int] = sizeClusterCol.indices.map(idx => List.fill(sizeClusterCol(idx))(idx)).reduce(_ ++ _)
        //Do the same per row cluster inside each column cluster
        //Each series in row is included in one cluster per column cluster
        val trueRowPartition: List[Int] = sizeClusterRow.indices.map(idx => List.fill(sizeClusterRow(idx))(idx)).reduce(_ ++ _)
        val trueColPartitionDF = ss.sparkContext.parallelize(trueColPartition.zipWithIndex.map(c => (c._2, c._1))).toDF("vName", "trueFeatureCluster")
        val trueRowPartitionDF = ss.sparkContext.parallelize(trueRowPartition.zipWithIndex.map(c => (c._2, c._1))).toDF("scid", "trueSeriesCluster")
        //Join true cluster values to TSS
        val tss2 = tss.addByLeftJoin(trueColPartitionDF, trueColPartitionDF("vName") === tss.getDecoratorColumn(TSS.VARNAME_DECORATORNAME)).drop("vName")
        tss2.addByLeftJoin(trueRowPartitionDF, trueRowPartitionDF("scid") === tss2.getDecoratorColumn(TSS.SIMULATIONID_DECORATORNAME)).drop("scid")
    }

    def generateDataPerBlock(modes: List[List[DenseVector[Double]]],
                           covariances: List[List[DenseMatrix[Double]]],
                           sizeClusterRow: List[List[Int]],
                           sizeClusterCol: List[Int],
                           mixingProportion: Double=0D): List[List[DenseMatrix[DenseVector[Double]]]]={

        require(mixingProportion>=0D & mixingProportion <=1D)
        val L = modes.length
        val KVec = modes.map(_.length)
        val MGaussians = (0 until L).map(l => {
            modes(l).indices.map(k => {
                MultivariateGaussian(modes(l)(k),covariances(l)(k))
            })
        })
        modes.indices.map(l => {
        val K_l = modes(l).length
        modes(l).indices.map(k => {
            val dataList: Array[DenseVector[Double]] = MGaussians(l)(k).sample(sizeClusterRow(l)(k)*sizeClusterCol(l)).toArray
            val mixingIndic = dataList.indices.map(_ => sample(List(1-mixingProportion, mixingProportion))).toList

            val mixedData = (dataList zip mixingIndic).map(c =>
            if(c._2==1){
                val newl = (l+1)%L
                val newk = (k+1)%KVec(newl)
                MGaussians(newl)(newk).draw()
            } else {
                c._1
            })
            DenseMatrix(mixedData).reshape(sizeClusterRow(l)(k),sizeClusterCol(l))
        }).toList
        }).toList
    }

    def doubleReduce(dataList: List[List[DenseMatrix[DenseVector[Double]]]]): DenseMatrix[DenseVector[Double]] = {
        dataList.indices.map(l => {
        dataList(l).indices.map(k_l => {
            dataList(l)(k_l)
        }).reduce(DenseMatrix.vertcat(_,_))
        }).reduce(DenseMatrix.horzcat(_,_))
    }

}