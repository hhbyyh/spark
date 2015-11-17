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

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}

object CNNExample {

  def main(args: Array[String]) {

    val myLayers = new Array[Layer](8)
    myLayers(0) = new ConvolutionalLayer(1, 6, new MapSize(5, 5), new MapSize(28, 28))
    myLayers(1) = new FunctionalLayer(new SigmoidFunction())
    myLayers(2) = new MeanPoolingLayer(new MapSize(2, 2), new MapSize(24, 24))
    myLayers(3) = new ConvolutionalLayer(6, 12, new MapSize(5, 5), new MapSize(12, 12))
    myLayers(4) = new FunctionalLayer(new SigmoidFunction())
    myLayers(5) = new MeanPoolingLayer(new MapSize(2, 2), new MapSize(8, 8))
    myLayers(6) = new ConvolutionalLayer(12, 12, new MapSize(4, 4), new MapSize(4, 4))
    myLayers(7) = new FunctionalLayer(new SigmoidFunction())
    val topology = FeedForwardTopology(myLayers)

    val conf = new SparkConf().setAppName("ttt")
    val sc = new SparkContext(conf)
    val data = MLUtils.loadLabeledPoints(sc, "data/mllib/sample_mnist_data.txt")
      .map { l =>
        val target = new Array[Double](12)
        target(l.label.toInt) = 1
        (l.features, Vectors.dense(target))
      }

    val feedForwardTrainer = new FeedForwardTrainer(topology, 784, 12)
    feedForwardTrainer.setStackSize(16)
      .SGDOptimizer
      .setMiniBatchFraction(0.5)
      .setNumIterations(500)

    val mlpModel = feedForwardTrainer.train(data)
    feedForwardTrainer.setWeights(mlpModel.weights())

    // predict
    // scalastyle:off println
    val right = data.filter(v => mlpModel.predict(v._1).argmax == v._2.argmax).count()
    val precision = right.toDouble / data.count()
    println(s"right: $right, count: ${data.count()}, precision: $precision")
    // scalastyle:on println
  }

}
