//
//  Extensions.swift
//  AZ
//
//  Created by Tanmay Bakshi on 2020-04-13.
//  Copyright © 2020 Tanmay Bakshi. All rights reserved.
//

import Foundation
import TensorFlow

func toTensor<T: TensorFlowScalar>(_ arr: [T]) -> Tensor<T> {
    return Tensor(arr)
}

func toTensor<T: TensorFlowScalar>(_ arr: [[T]]) -> Tensor<T> {
    return Tensor(arr.map({ toTensor($0) }))
}

func toTensor<T: TensorFlowScalar>(_ arr: [[[T]]]) -> Tensor<T> {
    return Tensor(arr.map({ toTensor($0) }))
}

extension Collection where Element: Comparable {
    func argmax() -> Int? {
        return self.enumerated().max { (a, b) -> Bool in
            a.element < b.element
        }?.offset
    }
}

func rotate90<T>(_ arr: [[T]], size: Int, dVal: T) -> [[T]] {
    var newArr = [[T]](repeating: [T](repeating: dVal, count: size), count: size)
    for i in 0..<size {
        for j in 0..<size {
            newArr[i][j] = arr[size - j - 1][i]
        }
    }
    return newArr
}

func rotate90<T>(_ arr: [T], size: Int, dVal: T) -> [T] {
    var newArr = [T](repeating: dVal, count: size*size)
    for i in 0..<size {
        for j in 0..<size {
            newArr[i + size * j] = arr[(size - j - 1) + size * i]
        }
    }
    return newArr
}

func flip<T>(_ arr: [[T]]) -> [[T]] {
    return arr.map({ $0.reversed() })
}

func flip1d<T>(_ arr: [T], size: Int) -> [T] {
    // Inefficient implementation
    var arr2d: [[T]] = []
    arr2d.reserveCapacity(size)
    for i in 0..<size {
        var row: [T] = []
        for j in 0..<size {
            row.append(arr[i + size * j])
        }
        arr2d.append(row)
    }
    arr2d = flip(arr2d)
    return arr2d.reduce([], +)
}

func *<T: Numeric>(lhs: [T], rhs: T) -> [T] {
    var newArray: [T] = []
    newArray.reserveCapacity(lhs.count)
    for i in 0..<lhs.count {
        newArray.append(lhs[i] * rhs)
    }
    return newArray
}

func *=<T: Numeric>(lhs: inout [T], rhs: [T]) {
    for i in 0..<lhs.count {
        lhs[i] *= rhs[i]
    }
}

func /=<T: FloatingPoint>(lhs: inout [T], rhs: T) {
    for i in 0..<lhs.count {
        lhs[i] /= rhs
    }
}

func /=<T: FloatingPoint>(lhs: inout [T], rhs: [T]) {
    for i in 0..<lhs.count {
        lhs[i] /= rhs[i]
    }
}

func random(probabilities: [Float]) -> Int {
    let chooseValue = Float.random(in: 0..<1)
    var accum: Float = 0
    for (index, value) in probabilities.enumerated() {
        accum += value
        if accum > chooseValue {
            return index
        }
    }
    return probabilities.count - 1
}
