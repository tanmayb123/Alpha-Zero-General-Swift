//
//  AZ.swift
//  AZ
//
//  Created by Tanmay Bakshi on 2020-04-13.
//  Copyright © 2020 Tanmay Bakshi. All rights reserved.
//

import Foundation

let EPS: Float = 1e-5

public protocol Game {
    associatedtype SimpleRepresentation: Hashable
    
    var initBoard: [[Float]] { get }
    var boardSize: (Int, Int) { get }
    var actionSize: Int { get }
    func getNextState(board: [[Float]], player: Int, action: Int) -> ([[Float]], Int)
    func getValidMoves(board: [[Float]], player: Int) -> [Float]
    func getGameEnded(board: [[Float]], player: Int) -> Float
    func getCanonicalForm(board: [[Float]], player: Int) -> [[Float]]
    func getSymmetries(board: [[Float]], pi: [Float]) -> ([[[Float]]], [[Float]])
    func simpleRepresentation(board: [[Float]]) -> SimpleRepresentation
}

public protocol NNet {
    associatedtype Task: Game
    
    init(game: Task)
    
    mutating func train(boards: [[[Float]]], pis: [[Float]], vs: [[Float]])
    func predict(board: [[Float]]) -> ([Float], Float)
    
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
                let (p, v) = nnet.predict(board: canonicalBoard)
                Ps[s] = p
                let valids = game.getValidMoves(board: canonicalBoard, player: 1)
                Ps[s]! *= valids
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

class Arena<Task: Game> {
    typealias Player = ([[Float]]) -> (Int)
    
    var player1: Player
    var player2: Player
    let game: Task
    
    init(player1: @escaping Player, player2: @escaping Player, game: Task) {
        self.player1 = player1
        self.player2 = player2
        self.game = game
    }
    
    func playGame() -> Float {
        let players = [1: player1, -1: player2]
        var currentPlayer = 1
        var board = game.initBoard
        while game.getGameEnded(board: board, player: currentPlayer) == 0 {
            let action = players[currentPlayer]!(game.getCanonicalForm(board: board, player: currentPlayer))
            let valids = game.getValidMoves(board: board, player: currentPlayer)
            if valids[action] == 0 {
                fatalError()
            }
            (board, currentPlayer) = game.getNextState(board: board, player: currentPlayer, action: action)
        }
        return Float(currentPlayer) * game.getGameEnded(board: board, player: currentPlayer)
    }
    
    func playGames(num: Int) -> (Int, Int, Int) {
        var (wins, draws, losses) = (0, 0, 0)
        for game in 1...num {
            print("Arena: playing game \(game) of \(num) (1v2)")
            let result = playGame()
            if result == 1 {
                wins += 1
            } else if result == -1 {
                losses += 1
            } else {
                draws += 1
            }
        }
        for game in 1...num {
            print("Arena: playing game \(game) of \(num) (2v1)")
            let result = playGame()
            if result == -1 {
                wins += 1
            } else if result == 1 {
                losses += 1
            } else {
                draws += 1
            }
        }
        print("Arena: finished.")
        return (wins, draws, losses)
    }
}

public class Coach<Task: Game, Network: NNet> where Network.Task == Task {
    struct TrainExamples {
        var boards: [[[Float]]]
        var pis: [[Float]]
        var vs: [[Float]]
    }
    
    private var game: Task
    private var nnet: Network
    private var pnet: Network
    
    private let historyAccessSemaphore = DispatchSemaphore(value: 1)
    private var trainExamplesHistory: [TrainExamples] = []
    
    var mctsSims = 25
    var tempThreshold = 15
    var iters = 100
    var eps = 100
    var historyThreshold = 20
    var arenaGames = 20
    var updateThreshold: Float = 0.6
    
    init(game: Task, nnet: Network) {
        self.game = game
        self.nnet = nnet
        self.pnet = Network(game: game)
    }
    
    func executeEpisode() -> TrainExamples {
        var mcts = MCTS(game: game, nnet: nnet, numSims: mctsSims)
        
        var trainExamples: [([[Float]], Int, [Float])] = []
        var board = game.initBoard
        var currentPlayer: Int = 1
        var episodeStep = 0
        
        while true {
            episodeStep += 1
            let canonicalBoard = game.getCanonicalForm(board: board, player: currentPlayer)
            let temp: Float = episodeStep < tempThreshold ? 1 : 0
            
            let pi = mcts.getActionProbs(canonicalBoard: canonicalBoard, temperature: temp)
            let (boardSyms, piSyms) = game.getSymmetries(board: canonicalBoard, pi: pi)
            for val in zip(boardSyms, piSyms) {
                trainExamples.append((val.0, currentPlayer, val.1))
            }
            
            let action = random(probabilities: pi)
            (board, currentPlayer) = game.getNextState(board: board, player: currentPlayer, action: action)
            
            let r = game.getGameEnded(board: board, player: currentPlayer)
            
            if r != 0 {
                let boards = trainExamples.map({ $0.0 })
                let pis = trainExamples.map({ $0.2 })
                let vs = trainExamples.map({ [Float]([r * Float($0.1 != currentPlayer ? -1 : 1)]) })
                return .init(boards: boards, pis: pis, vs: vs)
            }
        }
    }
    
    func trainingData() -> TrainExamples {
        var data = TrainExamples(boards: [], pis: [], vs: [])
        for iterData in trainExamplesHistory {
            data.boards += iterData.boards
            data.pis += iterData.pis
            data.vs += iterData.vs
        }
        let examples = Array(0..<data.boards.count).shuffled()
        data.boards = examples.map({ data.boards[$0] })
        data.pis = examples.map({ data.pis[$0] })
        data.vs = examples.map({ data.vs[$0] })
        return data
    }

    func learn() {
        for iter in 1...iters {
            print("Iter \(iter)")
            
            let completionSemaphore = DispatchSemaphore(value: 0)
            
            let accessSemaphore = DispatchSemaphore(value: 1)
            var episodesData: TrainExamples = .init(boards: [], pis: [], vs: [])
            DispatchQueue.global().async {
                let spawnSemaphore = DispatchSemaphore(value: 35)
                for episode in 1...self.eps {
                    spawnSemaphore.wait()
                    DispatchQueue.global().async {
                        defer {
                            completionSemaphore.signal()
                            spawnSemaphore.signal()
                        }
                        
                        let result = self.executeEpisode()
                        accessSemaphore.wait()
                        episodesData.boards += result.boards
                        episodesData.pis += result.pis
                        episodesData.vs += result.vs
                        accessSemaphore.signal()
                    }
                }
            }
            for episode in 1...eps {
                completionSemaphore.wait()
                print("Done episode \(episode)")
            }
            trainExamplesHistory.append(episodesData)
            
            if trainExamplesHistory.count > historyThreshold {
                trainExamplesHistory.removeFirst()
            }
            
            let data = trainingData()
            
            nnet.save(checkpoint: "temp.h5")
            pnet.load(checkpoint: "temp.h5")
            var pmcts = MCTS(game: game, nnet: pnet, numSims: mctsSims)
            
            nnet.train(boards: data.boards, pis: data.pis, vs: data.vs)
            var nmcts = MCTS(game: game, nnet: nnet, numSims: mctsSims)
            
            print("PITTING AGAINST PREVIOUS VERSION")
            
            let arena = Arena(player1: { pmcts.getActionProbs(canonicalBoard: $0, temperature: 0).argmax()! }, player2: { nmcts.getActionProbs(canonicalBoard: $0, temperature: 0).argmax()! }, game: game)
            let (pwins, draws, nwins) = arena.playGames(num: arenaGames)
            
            print("NEW/PREV WINS: \(nwins) / \(pwins) ; DRAWS: \(draws)")
            
            if pwins + nwins == 0 || Float(nwins) / Float(pwins + nwins) < updateThreshold {
                print("REJECTING NEW MODEL")
                nnet.load(checkpoint: "temp.h5")
            } else {
                print("ACCEPTING NEW MODEL")
                nnet.save(checkpoint: "best\(iter).h5")
            }
        }
    }
}
