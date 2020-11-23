package com.github.unsupervise.spark.tss.core

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.spark.sql.api.java.UDF1
import org.apache.spark.sql.{Column, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.{collect_list, struct, udf}
import org.apache.spark.sql.types._
import smile.mds

case class GroupedDistance(refId: Long, distances: Seq[(Long, Double)])

/**
  * Created by Anthony Coutant on 25/01/2019.
  */
//TODO: Generalize with multiple distances per pair (for easyness of use)
//For now, only unimodal (distances inside one set of objects) data is considered
//Dimension is given as parameter for computation speedup (avoid computing costly distinct count)
/**
  * TSDS class stands for Time Series Distances Set and is the base class for pairwise time series distances.
  * It is a wrapper around a Dataset[(Int, Int, Double)] object evolving accross TS parallel pipelines
  * @param indexedDistances the inner distances Dataset
  * @param dim the number of source elements compared (the size of domain of each id1 and id2 columns). Avoids a "distinct count" computation
  */
case class TSDS(indexedDistances: Dataset[IndexedDistance], dim: Int) {

  import indexedDistances.sparkSession.implicits._

  lazy val count = indexedDistances.count()

  val distancesSchema = StructType(Array(StructField(TSDS.ID1_COLNAME, LongType, false), StructField(TSDS.DISTANCE_COLNAME, DoubleType, false)))
  val sortById: Seq[Row] => Seq[Row] = _.sortBy(_.getLong(0))
  val sortByDistance: Seq[Row] => Seq[Row] = _.sortBy(_.getDouble(1))
  val getId: Seq[Row] => Seq[Long] = _.map(_.getLong(0))
  val getDistance: Seq[Row] => Seq[Double] = _.map(_.getDouble(1))

  /**
    * UDF to sort distances by first element of a pair column
    * Used for example to group distances by id1, sorted by id2, with collect_list agg
    */
  val sortByIdUDF = udf((distances: Seq[Row]) => {
    sortByDistance(distances)
  }, distancesSchema)
  /**
    * UDF to sort distances by second element of a pair column
    * Used for example to group distances by distance, sorted by id2, with collect_list agg
    */
  val sortByDistanceUDF = udf((distances: Seq[Row]) => {
    sortById(distances)
  }, distancesSchema)
  /**
    * UDF to map (id, distance) pairs to id element
    * Used for example to group distances by id1, sorted by id2, with collect_list agg
    */
  val getIdUDF = udf((distances: Seq[Row]) => {
    getId(distances)
  })
  /**
    * UDF to map (id, distance) pairs to distance element
    * Used for example to group distances by id1, sorted by id2, with collect_list agg
    */
  val getDistanceUDF = udf((distances: Seq[Row]) => {
    getDistance(distances)
  })
  /**
    * UDF to sort distances by first id element of a pair column then keep second distance element
    * Used for example to group distances by id1, sorted by id2, with collect_list agg
    */
  val sortByIdGetDistanceUDF = udf((distances: Seq[Row]) => {
    getDistance(sortById(distances))
  })
  /**
    * UDF to sort distances by second distance element of a pair column then keep first id element
    * Used for example to group distances by id1, sorted by distance, with collect_list agg
    */
  val sortByDistanceGetIdUDF = udf((distances: Seq[Row]) => {
    getId(sortByDistance(distances))
  })

  /**
    * Compute the TSDS wrapper of ordered inner distances by (id1, id2). Useful to generate a Double matrix
    * in an adequate order.
    * @return the TSDS wrapper around ordered distances Dataset.
    */
  def orderByIds = TSDS(indexedDistances.orderBy(TSDS.ID1_COLNAME, TSDS.ID2_COLNAME), dim)

  /**
    * Select distances
    * @return inner distances restricted to Double distances column
    */
  def selectDistances = indexedDistances.select(TSDS.DISTANCE_COLNAME)

  /**
    * Collect distances column
    * @return inner distances distances column collected as Array[Double]
    */
  def collectDistances = selectDistances.map(_.getDouble(0)).collect()

  /**
    * Collect distances grouped by id1 and ordered by (id1, id2) as an Array[Array[Double]].
    * Mainly used for input to Scala SMILE library
    * @return the ordered Array[Array[Double]]
    */
  def getAs2DArray = {

    orderByIds.collectDistances.grouped(dim).toArray

    /*val collectedDists = orderByIds.collectDistances.grouped(dim).toArray
    //Since distances are ordered by ids, cut the collected
    //sequence as blocks of uniqueId2Count values
    val slices = (0 until collectedDists.length by dim).map(i => {
      collectedDists.slice(i, i + dim)
    })
    slices.toArray*/
  }

  def getAggregateById(aggreg: Column, groupColName: String = TSDS.ID1_COLNAME, orderByColName: String = TSDS.ID2_COLNAME) = {
    assert(groupColName.equals(TSDS.ID1_COLNAME) || groupColName.equals(TSDS.ID2_COLNAME))
    assert(orderByColName.equals(TSDS.ID1_COLNAME) || groupColName.equals(TSDS.ID2_COLNAME) || groupColName.equals(TSDS.DISTANCE_COLNAME))
    assert(!orderByColName.equals(groupColName))
    //val otherColName = if(groupColName.equals(TSDS.ID1_COLNAME)) TSDS.ID2_COLNAME else TSDS.ID1_COLNAME
    indexedDistances
      .orderBy(indexedDistances(orderByColName))
      .groupBy(indexedDistances(groupColName).alias("refId"))
      //.agg(collect_list(struct(otherColName, TSDS.DISTANCE_COLNAME)).alias("distances"))
      .agg(aggreg)
      //.as[GroupedDistance]
  }

  /**
    * Get distances as grouped list List[(Int, Array[Double])] by either id1 or id2
    * the remaining id being stored in the List of pairs.
    * Beware: the id stored in the List of pairs is NOT SORTED by default!
    * @param groupColName the column name to use for grouping (must be either id1 or id2)
    * @return the DataFrame with 2 columns: refId (id1 or id2) and distances (with every id2 if refId is id1, or id1 if refId is id2)
    */
  def getAsGroupedLists(groupColName: String = TSDS.ID1_COLNAME) = {
    assert(groupColName.equals(TSDS.ID1_COLNAME) || groupColName.equals(TSDS.ID2_COLNAME))
    val orderedDistances = orderByIds
    val otherColName = if(groupColName.equals(TSDS.ID1_COLNAME)) TSDS.ID2_COLNAME else TSDS.ID1_COLNAME
    orderedDistances
      .groupBy(orderedDistances(groupColName).alias("refId"))
      .agg(collect_list(struct(otherColName, TSDS.DISTANCE_COLNAME)).alias("distances"))
      .as[GroupedDistance]
  }

  /**
    * Proxy of Spark group by function
    * @param columns the group by Column parameters
    * @return the grouped by DataFrame
    */
  def groupBy(columns: Column*) = indexedDistances.groupBy(columns: _*)

  /**
    * Proxy of Spark collect function
    * @return the collected distances, as Array[Row], each Row having the 3 dimensions id1, id2, distance
    */
  def collect = indexedDistances.collect()

  /**
    * Save inner distances DataFrame as com.databricks.spark.csv file
    * @param path the output chunked csv file path
    * @param header whether to add header to output file
    * @param overwrite whether to automatically overwrite existing output file
    */
  def saveCSV(path: String, header: Boolean = true, overwrite: Boolean = false) = {
    indexedDistances.write.format("com.databricks.spark.csv").option("header", header).save(path)
  }

  /**
    * Helper to provide Spark like parenthesis support to retrieve a Column from a column name
    * @param colname the column name to retrieve as a Column object
    * @return the request column name as a Column
    */
  def apply(colname: String) = indexedDistances(colname)

}

object TSDS {

  val ID1_COLNAME = "id1"
  val ID2_COLNAME = "id2"
  val DISTANCE_COLNAME = "distance"

}