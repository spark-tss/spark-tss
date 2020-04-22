package com.github.unsupervise.spark.tss.core

/**
  * Created by antho on 19/12/2018.
  */

/**
  * Case class representing a distance between two indexed items
  * @param id1 the index of first item
  * @param id2 the index of second item
  * @param distance the distance value between both items
  */
case class IndexedDistance(id1: Long, id2: Long, distance: Double)
