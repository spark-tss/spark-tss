package com.github.unsupervise.spark.tss.core

/**
  * Created by Anthony Coutant on 25/01/2019.
  */

/**
  * Case class for distance indexed with only one id (used for selections after group by statements)
  * @param id2 the index
  * @param distance the distance value
  */
case class SemiIndexedDistance(id2: Long, distance: Double)
