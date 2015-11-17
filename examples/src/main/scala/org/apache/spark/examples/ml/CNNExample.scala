package org.apache.spark.ml.ann

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}

object CNNExample {

  def main(args: Array[String]) {

    val myLayers = new Array[Layer](8)
    myLayers(0) = new ConvolutionalLayer(1, 6, kernelSize = new MapSize(5, 5), inputMapSize = new MapSize(28, 28))
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
    val right = data.filter(v => mlpModel.predict(v._1).argmax == v._2.argmax).count()
    val precision = right.toDouble / data.count()
    println(s"right: $right, count: ${data.count()}, precision: $precision")
  }

}
