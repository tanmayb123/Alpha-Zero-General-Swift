//
//  main.swift
//  AZ
//
//  Created by Tanmay Bakshi on 2020-04-13.
//  Copyright © 2020 Tanmay Bakshi. All rights reserved.
//

import Foundation
import PythonKit
PythonLibrary.useVersion(3, 6)

let game = OthelloGame(n: 8)
var nnet = OthelloNNet(game: game)
var coach = Coach(game: game, nnet: nnet)
coach.learn()
