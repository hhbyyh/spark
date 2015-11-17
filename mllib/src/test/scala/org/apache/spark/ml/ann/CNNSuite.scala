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

package org.apache.spark.ml.ann

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.util.MLlibTestSparkContext
import breeze.linalg.{DenseMatrix => BDM}

class CNNSuite extends SparkFunSuite with MLlibTestSparkContext {

  test("Conv valid") {
    val m = new BDM[Double](4, 4, (1 to 16).map(_.toDouble).toArray)
    val kernel = new BDM[Double](2, 2, (1 to 4).map(_.toDouble).toArray)
    val result = ConvolutionalLayerModel.convValid(m, kernel)
    assert(Array(44.0, 54.0, 64.0, 84.0, 94.0, 104.0, 124.0, 134.0, 144.0)
      .zip(result.toArray)
      .forall(p => p._1 == p._2)
    )
  }

  test("Conv full") {
    val m = new BDM[Double](3, 3, (1 to 9).map(_.toDouble).toArray)
    val kernel = new BDM[Double](2, 2, (1 to 4).map(_.toDouble).toArray)
    val result = ConvolutionalLayerModel.convFull(m, kernel)
    assert(Array(4.0, 3.0, 0, 0, 18.0, 13.0, 0.0, 0.0, 36.0, 25.0, 0.0, 0.0, 14.0, 7.0, 0.0, 0.0)
      .zip(result.toArray)
      .forall(p => p._1 == p._2)
    )
  }

  test("mean pooling") {
    val m = new BDM[Double](4, 4, (1 to 16).map(_.toDouble).toArray)
    val result = MeanPoolingLayerModel.avgPooling(m, new MapSize(2, 2))
    assert(Array(3.5, 5.5, 11.5, 13.5)
      .zip(result.toArray)
      .forall(p => p._1 == p._2)
    )
  }

}
