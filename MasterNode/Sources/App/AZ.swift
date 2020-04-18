//
//  AZ.swift
//  AZ
//
//  Created by Tanmay Bakshi on 2020-04-13.
//  Copyright © 2020 Tanmay Bakshi. All rights reserved.
//

import Foundation
import PythonKit
import Alamofire

let EPS: Float = 1e-5
let WORKERS = ["testworker.ngrok.io"].map { "http://\($0):8080/" }

public protocol Game {
    associatedtype SimpleRepresentation: Hashable
    
    var initBoard: [[Float]] { get }
    var boardSize: (Int, Int) { get }
    var actionSize: Int { get }
    func getNextState(board: [[Float]], player: Int, action: Int) -> ([[Float]], Int)
    func getValidMoves(board: [[Float]], player: Int) -> [Float]
    func getGameEnded(board: [[Float]], player: Int) -> Float
    func getCanonicalForm(board: [[Float]], player: Int) -> [[Float]]
    func getSymmetries(board: [[Float]], pi: [Float], valids: [Float]) -> ([[[Float]]], [[Float]], [[Float]])
    func simpleRepresentation(board: [[Float]]) -> SimpleRepresentation
}

public protocol NNet {
    associatedtype Task: Game
    
    init(game: Task)
    
    mutating func train()
    func predict(board: [[Float]], valids: [Float]) -> ([Float], Float)

    mutating func storeEpisode(boards: [[[Float]]], pis: [[Float]], vs: [[Float]], valids: [[Float]])
    
    func save(checkpoint: String)
    mutating func load(checkpoint: String)
}

public struct MCTS<Task: Game, Network: NNet> {
    struct SA: Hashable {
        var state: Task.SimpleRepresentation
        var action: Int
        
        init(_ state: Task.SimpleRepresentation, _ action: Int) {
            self.state = state
            self.action = action
        }
    }
    
    var game: Task
    var nnet: Network
    var numSims: Int
    
    private var Qsa: [SA: Float] = [:]
    private var Nsa: [SA: Float] = [:]
    private var Ns: [Task.SimpleRepresentation: Float] = [:]
    private var Ps: [Task.SimpleRepresentation: [Float]] = [:]
    
    private var Es: [Task.SimpleRepresentation: Float] = [:]
    private var Vs: [Task.SimpleRepresentation: [Float]] = [:]

    init(game: Task, nnet: Network, numSims: Int) {
        self.game = game
        self.nnet = nnet
        self.numSims = numSims
    }
    
    mutating func getActionProbs(canonicalBoard: [[Float]], temperature: Float) -> [Float] {
        for _ in 0..<numSims {
            search(canonicalBoard: canonicalBoard)
        }
        
        let s = game.simpleRepresentation(board: canonicalBoard)
        var counts = (0..<game.actionSize).map { a -> Float in
            Nsa[SA(s, a), default: 0]
        }
        
        if temperature == 0 {
            let bestA = counts.argmax()!
            var probs = [Float](repeating: 0, count: game.actionSize)
            probs[bestA] = 1
            return probs
        }
        
        counts = counts.map({ pow($0, 1 / temperature) })
        let countsSum = counts.reduce(0, +)
        let probs = counts.map({ $0 / countsSum })
        return probs
    }

    mutating func search(canonicalBoard: [[Float]]) {
        var boards: [[[Float]]] = [canonicalBoard]
        
        var vsa: [(Float?, SA?)] = []
        
        while let canonicalBoard = boards.first {
            defer { boards.removeFirst() }
            
            let s = game.simpleRepresentation(board: canonicalBoard)
                    
            if !Es.keys.contains(s) {
                Es[s] = game.getGameEnded(board: canonicalBoard, player: 1)
            }
            if Es[s]! != 0 {
                vsa.append((-Es[s]!, nil))
                continue
            }
            
            guard Ps.keys.contains(s) else {
                let valids = game.getValidMoves(board: canonicalBoard, player: 1)
                let (p, v) = nnet.predict(board: canonicalBoard, valids: valids)
                Ps[s] = p
                let sumP = Ps[s]!.reduce(0, +)
                if sumP > 0 {
    //                print("Not all masked")
                    Ps[s]! /= sumP
                } else {
    //                print("All valid moves were masked, do workaround.")
                    Ps[s]! += valids
                    Ps[s]! /= Ps[s]!.reduce(0, +)
                }
                Vs[s] = valids
                Ns[s] = 0
                vsa.append((-v, nil))
                continue
            }
            
            let valids = Vs[s]!
            var currentBest = -Float.infinity
            var bestAct = -1
            
            for a in 0..<game.actionSize {
                if valids[a] == 1 {
                    let u: Float
                    if Qsa.keys.contains(SA(s, a)) {
                        u = Qsa[SA(s, a)]! + 1 * Ps[s]![a] * sqrt(Ns[s]!) / (1 + Nsa[SA(s, a)]!)
                    } else {
                        u = 1 * Ps[s]![a] * sqrt(Ns[s]! + EPS)
                    }
                    
                    if u > currentBest {
                        currentBest = u
                        bestAct = a
                    }
                }
            }
            
            let a = bestAct
            let (nextState, nextPlayer) = game.getNextState(board: canonicalBoard, player: 1, action: a)
            let canonicalNextState = game.getCanonicalForm(board: nextState, player: nextPlayer)
            
            boards.append(canonicalNextState)
                        
            vsa.append((nil, SA(s, a)))
        }
        
        var currentV = vsa.removeLast().0!
        for i in (0..<vsa.count).reversed() {
            if let v = vsa[i].0 {
                currentV = -v
            } else {
                if let sa = vsa[i].1 {
                    if Qsa.keys.contains(sa) {
                        Qsa[sa] = (Nsa[sa]! * Qsa[sa]! + currentV) / (Nsa[sa]! + 1)
                        Nsa[sa]! += 1
                    } else {
                        Qsa[sa] = currentV
                        Nsa[sa] = 1
                    }
                }
            }
        }
    }
}

public class DistributedCoach<Task: Game, Network: NNet> where Network.Task == Task {
    struct TrainExamples: Codable {
        var id: Int = -1
        var boards: [[[Float]]]
        var pis: [[Float]]
        var vs: [[Float]]
        var valids: [[Float]]

        mutating func add(_ other: TrainExamples) {
            if id != other.id {
                fatalError()
            }
            boards += other.boards
            pis += other.pis
            vs += other.vs
            valids += other.valids
        }

        mutating func reset() {
            boards = []
            pis = []
            vs = []
            valids = []
        }
    }
    
    private var game: Task
    private var nnet: Network
    
    var mctsSims = 40
    var tempThreshold = 15
    var iters = 500
    var eps = 2500

    var genId = -1
    var totalEpisodes = 0
    var iterationExamples: TrainExamples = .init(boards: [], pis: [], vs: [], valids: [])
    let iterationExamplesAccessSemaphore = DispatchSemaphore(value: 1)
    let iterationCompleteSemaphore = DispatchSemaphore(value: 0)
    
    init(game: Task, nnet: Network) {
        self.game = game
        self.nnet = nnet
    }

    func sendNetworkToWorkers(id: String) {
        for worker in WORKERS {
            _ = AF.request(worker + "getNetwork", method: .get, parameters: ["id": id])
        }
    }

    func launchWorkers() {
        genId = (Int.min...Int.max).randomElement()!
        for worker in WORKERS {
            _ = AF.request(worker + "generate", method: .get, parameters: ["id": genId])
        }
    }

    func stopWorkers() {
        for worker in WORKERS {
            _ = AF.request(worker + "stop", method: .get)
        }
    }
    
    func gotEpisode(data: TrainExamples) {
        iterationExamplesAccessSemaphore.wait()
        defer { iterationExamplesAccessSemaphore.signal() }
        guard data.id != genId else {
            return
        }
        totalEpisodes += 1
        iterationExamples.add(data)
        if totalEpisodes >= eps {
            iterationCompleteSemaphore.signal()
        }
    }
   
    func learn() {
        nnet.save(checkpoint: "masterinit.h5")
        sendNetworkToWorkers(id: "init")

        for iter in 1...iters {
            print("Iter \(iter)")

            totalEpisodes = 0
            iterationExamples.reset()

            launchWorkers()
            iterationCompleteSemaphore.wait()
            stopWorkers()

            iterationExamplesAccessSemaphore.wait()
            
            nnet.storeEpisode(boards: iterationExamples.boards, pis: iterationExamples.pis, vs: iterationExamples.vs, valids: iterationExamples.valids)
            nnet.train()

            nnet.save(checkpoint: "masteriter\(iter).h5")
            sendNetworkToWorkers(id: "\(iter)")
        }
    }
}
