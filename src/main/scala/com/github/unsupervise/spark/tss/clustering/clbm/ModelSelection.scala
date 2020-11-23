/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.clbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import com.github.unsupervise.spark.tss.clustering.lbm.{LatentBlockModel}
import com.github.unsupervise.spark.tss.clustering.lbm
import breeze.linalg.{DenseVector, argmax}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object ModelSelection {

  def updateModel(formerModel: CondLatentBlockModel,
                  colClusterToUpdate: Int,
                  modelToInsert:CondLatentBlockModel): CondLatentBlockModel = {

    require(colClusterToUpdate >= 0 & colClusterToUpdate<formerModel.proportionsCols.length,
      "Col Cluster Idx to replace should be > 0 and lesser than the column cluster number of former model")

    val newRowProportion = remove(formerModel.proportionsRows, colClusterToUpdate) ++ modelToInsert.proportionsRows
    val newColProportion = remove(formerModel.proportionsCols, colClusterToUpdate) ++
      modelToInsert.proportionsCols.map(c => c*formerModel.proportionsCols(colClusterToUpdate))
    val newModels = remove(formerModel.gaussians, colClusterToUpdate) ++ modelToInsert.gaussians

    CondLatentBlockModel(newRowProportion, newColProportion, newModels)
  }

  def gridSearch(data: RDD[(Int, Array[DenseVector[Double]])],
                 rangeK:  List[Int],
                 rangeL: List[Int],
                 nConcurrent: Int = 3,
                 nTryMaxPerConcurrent: Int = 20,
                 initMethod: String = "random",
                 verbose: Boolean = false)(implicit ss: SparkSession): Map[String,Product] = {

    var everyCombinations: List[List[Int]] = List.empty[List[Int]]
    for(i <- rangeK){
      for(j <- rangeL){
        everyCombinations = everyCombinations++generateCombinationWithReplacement(i,j)}
    }
    everyCombinations = everyCombinations.distinct
    if(verbose){println("Launching "+everyCombinations.length.toString+" Combinations")}
    val everyRes:List[Map[String, Product]] = everyCombinations.map(combinationKVec => {
      if(verbose){println(">>> ", combinationKVec)}
      val CLBM = new CondLatentBlock().setKVec(combinationKVec)
      CLBM.run(data,
        nConcurrent=nConcurrent,
        nTryMaxPerConcurrent=nTryMaxPerConcurrent,
        initMethod=initMethod,
        verbose=verbose)
    })
    val everyLoglikelihood: DenseVector[Double] = DenseVector(everyRes.map(c => c("LogLikelihood").asInstanceOf[List[Double]].last).toArray)
    val everyICL: DenseVector[Double] = DenseVector(everyRes.map(c => c("ICL").asInstanceOf[List[Double]].last).toArray)
    if(verbose){everyICL.toArray.indices.foreach(i => {println(everyCombinations(i), "Loglikelihood: "+everyLoglikelihood(i).toString, "ICL: "+everyICL(i).toString)})}
    everyRes(argmax(everyICL))
  }

  def bestModelAfterLBMGridSearch(data: RDD[(Int, Array[DenseVector[Double]])],
                                  rangeL: List[Int],
                                  rangeK: List[Int],
                                  rangeKLBM: List[Int],
                                  EMMethod: String= "SEMGibbs",
                                  nConcurrentPerTry: Int = 5,
                                  verbose:Boolean = false,
                                  nTryMaxPerConcurrent:Int = 30)(implicit ss: SparkSession): Map[String,Product] = {

    val p:Int = data.take(1).head._2.length
    val n:Int = data.count().toInt

    val dataWithDummyRowPartition = data.map(r => (r._1, r._2, List(0)))

    val bestLBM = lbm.ModelSelection.gridSearch(
      data,
      rangeRow = rangeKLBM,
      rangeCol = rangeL,
      verbose= verbose,
      nConcurrentEachTest=nConcurrentPerTry,
      nTryMaxPerConcurrent=nTryMaxPerConcurrent)

    val LBMModel: LatentBlockModel = bestLBM("Model").asInstanceOf[LatentBlockModel]
    val bestL = LBMModel.L
    val CLBMModel = CondLatentBlockModel(
      List.fill(bestL){LBMModel.proportionsRows},
      LBMModel.proportionsCols,
      (0 until bestL).map(l => {
        LBMModel.proportionsRows.indices.map(k => {
          LBMModel.gaussians(l)(k)
        }).toList
      }).toList)

    val LBMColPartition = bestLBM("ColPartition").asInstanceOf[List[Int]]
    val dataWithRowPartition = CLBMModel.drawRowPartition(dataWithDummyRowPartition,LBMColPartition)
    var modelToUpdate = CLBMModel

    (0 until bestL).foreach(l => {
      if(verbose) {println("Updating component "+l.toString)}
      val currentColumnClusterData: RDD[(Int, Array[DenseVector[Double]])] = dataWithRowPartition.map(row =>
        (row._1, row._2.zipWithIndex.filter(s => LBMColPartition(s._2) == l).map(_._1)))

      val componentRes = ModelSelection.gridSearch(
        currentColumnClusterData,
        rangeK = rangeK,
        rangeL = List(1),
        nTryMaxPerConcurrent = 3,
        verbose=verbose)
      val componentModel = componentRes("Model").asInstanceOf[CondLatentBlockModel]
      // colClusterToUpdate is 0 because the last updated component has been put at the end of the component list.
      modelToUpdate = Tools.updateModel(modelToUpdate, 0, componentModel)
    })

    if(verbose){
      println(">>>>> Selected Model: ", modelToUpdate.KVec)}

    val globalDataWithRowPartition = modelToUpdate.drawRowPartition(dataWithRowPartition,LBMColPartition)

    val rowPartition = Tools.getRowPartitionFromDataWithRowPartition(globalDataWithRowPartition)

    val logLikelihood = modelToUpdate.completelogLikelihood(globalDataWithRowPartition,LBMColPartition)
    Map("Model" -> modelToUpdate,
      "RowPartition" -> rowPartition,
      "ColPartition" -> LBMColPartition,
      "LogLikelihood" -> List(logLikelihood),
      "ICL" -> List(modelToUpdate.ICL(logLikelihood,n,p,fullCovariance = true)))
  }

}
