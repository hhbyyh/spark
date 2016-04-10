/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.stat.test

import scala.annotation.varargs

import org.apache.commons.math3.distribution.{NormalDistribution, RealDistribution}
import org.apache.commons.math3.stat.inference.{KolmogorovSmirnovTest => CommonMathKolmogorovSmirnovTest}

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD

/**
 * Conduct the two-sided Kolmogorov Smirnov (KS) test for data sampled from a
 * continuous distribution. By comparing the largest difference between the empirical cumulative
 * distribution of the sample data and the theoretical distribution we can provide a test for the
 * the null hypothesis that the sample data comes from that theoretical distribution.
 * For more information on KS Test:
 * @see [[https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test]]
 *
 * Implementation note: We seek to implement the KS test with a minimal number of distributed
 * passes. We sort the RDD, and then perform the following operations on a per-partition basis:
 * calculate an empirical cumulative distribution value for each observation, and a theoretical
 * cumulative distribution value. We know the latter to be correct, while the former will be off by
 * a constant (how large the constant is depends on how many values precede it in other partitions).
 * However, given that this constant simply shifts the empirical CDF upwards, but doesn't
 * change its shape, and furthermore, that constant is the same within a given partition, we can
 * pick 2 values in each partition that can potentially resolve to the largest global distance.
 * Namely, we pick the minimum distance and the maximum distance. Additionally, we keep track of how
 * many elements are in each partition. Once these three values have been returned for every
 * partition, we can collect and operate locally. Locally, we can now adjust each distance by the
 * appropriate constant (the cumulative sum of number of elements in the prior partitions divided by
 * thedata set size). Finally, we take the maximum absolute value, and this is the statistic.
 */
private[stat] object KolmogorovSmirnovTest extends Logging {

  // Null hypothesis for the type of KS test to be included in the result.
  object NullHypothesis extends Enumeration {
    type NullHypothesis = Value
    val OneSampleTwoSided = Value("Sample follows theoretical distribution")
    val TwoSampleTwoSided = Value("Both samples follow same distribution")
  }

  /**
   * Runs a KS test for 1 set of sample data, comparing it to a theoretical distribution
   * @param data `RDD[Double]` data on which to run test
   * @param cdf `Double => Double` function to calculate the theoretical CDF
   * @return [[org.apache.spark.mllib.stat.test.KolmogorovSmirnovTestResult]] summarizing the test
   *        results (p-value, statistic, and null hypothesis)
   */
  def testOneSample(data: RDD[Double], cdf: Double => Double): KolmogorovSmirnovTestResult = {
    val n = data.count().toDouble
    val ksStat = data.sortBy(x => x).zipWithIndex().map { case (v, i) =>
      val f = cdf(v)
      math.max(f - i / n, (i + 1) / n - f)
    }.max()
    evalOneSampleP(ksStat, n.toLong)
  }

  /**
   * Runs a KS test for 1 set of sample data, comparing it to a theoretical distribution
   * @param data `RDD[Double]` data on which to run test
   * @param distObj `RealDistribution` a theoretical distribution
   * @return [[org.apache.spark.mllib.stat.test.KolmogorovSmirnovTestResult]] summarizing the test
   *        results (p-value, statistic, and null hypothesis)
   */
  def testOneSample(data: RDD[Double], distObj: RealDistribution): KolmogorovSmirnovTestResult = {
    val cdf = (x: Double) => distObj.cumulativeProbability(x)
    testOneSample(data, cdf)
  }

  /**
   * A convenience function that allows running the KS test for 1 set of sample data against
   * a named distribution
   * @param data the sample data that we wish to evaluate
   * @param distName the name of the theoretical distribution
   * @param params Variable length parameter for distribution's parameters
   * @return [[org.apache.spark.mllib.stat.test.KolmogorovSmirnovTestResult]] summarizing the
   *        test results (p-value, statistic, and null hypothesis)
   */
  @varargs
  def testOneSample(data: RDD[Double], distName: String, params: Double*)
    : KolmogorovSmirnovTestResult = {
    val distObj =
      distName match {
        case "norm" => {
          if (params.nonEmpty) {
            // parameters are passed, then can only be 2
            require(params.length == 2, "Normal distribution requires mean and standard " +
              "deviation as parameters")
            new NormalDistribution(params(0), params(1))
          } else {
            // if no parameters passed in initializes to standard normal
            logInfo("No parameters specified for normal distribution," +
              "initialized to standard normal (i.e. N(0, 1))")
            new NormalDistribution(0, 1)
          }
        }
        case  _ => throw new UnsupportedOperationException(s"$distName not yet supported through" +
          s" convenience method. Current options are:['norm'].")
      }

    testOneSample(data, distObj)
  }

  private def evalOneSampleP(ksStat: Double, n: Long): KolmogorovSmirnovTestResult = {
    val pval = 1 - new CommonMathKolmogorovSmirnovTest().cdf(ksStat, n.toInt)
    new KolmogorovSmirnovTestResult(pval, ksStat, NullHypothesis.OneSampleTwoSided.toString)
  }

    /**
   * Implements a two-sample, two-sided Kolmogorov-Smirnov test, which tests if the 2 samples
   * come from the same distribution
   * @param data1 `RDD[Double]` first sample of data
   * @param data2 `RDD[Double]` second sample of data
   * @return [[org.apache.spark.mllib.stat.test.KolmogorovSmirnovTestResult]] with the test
   *        statistic, p-value, and appropriate null hypothesis
   */
  def testTwoSamples(data1: RDD[Double], data2: RDD[Double]): KolmogorovSmirnovTestResult = {
    val n1 = data1.count().toDouble
    val n2 = data2.count().toDouble

    data1.sortBy(x => x).zipWithIndex().reduceByKey((num1, num2) => math.max(num1, num2))

    // identifier for sample 1, needed after co-sort
    val isSample1 = true
    // combine identified samples
    val unionedData = data1.map((_, isSample1)).union(data2.map((_, !isSample1)))
    // co-sort and operate on each partition, returning local extrema to the driver
    val localData = unionedData.sortByKey().mapPartitions(
      searchTwoSampleCandidates(_, n1, n2)
    ).collect()
    // result: global extreme
    val ksStat = searchTwoSampleStatistic(localData, n1 * n2)
    evalTwoSampleP(ksStat, n1.toInt, n2.toInt)
  }

   /**
   * Calculates maximum distance candidates and counts of elements from each sample within one
   * partition for the two-sample, two-sided Kolmogorov-Smirnov test implementation. Function
   * is package private for testing convenience.
   * @param partData `Iterator[(Double, Boolean)]` the data in 1 partition of the co-sorted RDDs,
   *                each element is additionally tagged with a boolean flag for sample 1 membership
   * @param n1 `Double` sample 1 size
   * @param n2 `Double` sample 2 size
   * @return `Iterator[(Double, Double, Double)]` where the first element is an unadjusted minimum
   *        distance, the second is an unadjusted maximum distance (both of which will later
   *        be adjusted by a constant to account for elements in prior partitions), and the third is
   *        a count corresponding to the numerator of the adjustment constant coming from this
   *        partition. This last value, the numerator of the adjustment constant, is calculated as
   *        |sample 2| * |sample 1 in partition| - |sample 1| * |sample 2 in partition|. This comes
   *        from the fact that when we adjust for prior partitions, what we are doing is
   *        adding the difference of the fractions (|prior elements in sample 1| / |sample 1| -
   *        |prior elements in sample 2| / |sample 2|). We simply keep track of the numerator
   *        portion that is attributable to each partition so that following partitions can
   *        use it to cumulatively adjust their values.
   */
  private[stat] def searchTwoSampleCandidates(
      partData: Iterator[(Double, Boolean)],
      n1: Double,
      n2: Double): Iterator[(Double, Double, Double)] = {
    // fold accumulator: local minimum, local maximum, index for sample 1, index for sample2
    case class ExtremaAndRunningIndices(min: Double, max: Double, ix1: Int, ix2: Int)
    val initAcc = ExtremaAndRunningIndices(Double.MaxValue, Double.MinValue, 0, 0)
    // traverse the data in the partition and calculate distances and counts
    val pResults = partData.foldLeft(initAcc) { case (acc, (v, isSample1)) =>
      val (add1, add2) = if (isSample1) (1, 0) else (0, 1)
      val cdf1 = (acc.ix1 + add1) / n1
      val cdf2 = (acc.ix2 + add2) / n2
      val dist = cdf1 - cdf2
      ExtremaAndRunningIndices(
        math.min(acc.min, dist),
        math.max(acc.max, dist),
        acc.ix1 + add1, acc.ix2 + add2
      )
    }
    // If partition has no data, then pResults will match the fold accumulator
    // we must filter this out to avoid having the statistic spoiled by the accumulation values
    val results = if (pResults == initAcc) {
      Array[(Double, Double, Double)]()
    } else {
      Array((pResults.min, pResults.max, (pResults.ix1 + 1) * n2 - (pResults.ix2 + 1) * n1))
    }
    results.iterator
  }

  /**
   * Adjust candidate extremes by the appropriate constant. The resulting maximum corresponds to
   * the two-sample, two-sided Kolmogorov-Smirnov test. Function is package private for testing
   * convenience.
   * @param localData `Array[(Double, Double, Double)]` contains the candidate extremes from each
   *                 partition, along with the numerator for the necessary constant adjustments
   * @param n `Double` The denominator in the constant adjustment (i.e. (size of sample 1 ) * (size
   *         of sample 2))
   * @return The two-sample, two-sided Kolmogorov-Smirnov statistic
   */
  private[stat] def searchTwoSampleStatistic(localData: Array[(Double, Double, Double)], n: Double)
    : Double = {
    // maximum distance and numerator for constant adjustment
    val initAcc = (Double.MinValue, 0.0)
    // adjust differences based on the number of elements preceding it, which should provide
    // the correct distance between the 2 empirical CDFs
    val results = localData.foldLeft(initAcc) { case ((prevMax, prevCt), (minCand, maxCand, ct)) =>
      val adjConst = prevCt / n
      val dist1 = math.abs(minCand + adjConst)
      val dist2 = math.abs(maxCand + adjConst)
      val maxVal = Array(prevMax, dist1, dist2).max
      (maxVal, prevCt + ct)
    }
    results._1
  }

  private def evalTwoSampleP(ksStat: Double, n: Int, m: Int): KolmogorovSmirnovTestResult = {
    val pval = new CommonMathKolmogorovSmirnovTest().approximateP(ksStat, n, m)
    new KolmogorovSmirnovTestResult(pval, ksStat, NullHypothesis.TwoSampleTwoSided.toString)
  }
}

