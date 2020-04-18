//
//  Othello.swift
//  AZ
//
//  Created by Tanmay Bakshi on 2020-04-14.
//  Copyright © 2020 Tanmay Bakshi. All rights reserved.
//

import Foundation
import TensorFlow
import PythonKit

let OTHELLO_PATH = "/home/tajymany/othello"

let sys = Python.import("sys")

struct OthelloLogic {
    struct Coord: Hashable {
        var x: Int
        var y: Int
        
        init(_ x: Int, _ y: Int) {
            self.x = x
            self.y = y
        }
        
        mutating func increment(_ dir: (Int, Int)) {
            x += dir.0
            y += dir.1
        }
        
        func isValid(max: Int) -> Bool {
            return 0 <= x && x < max && 0 <= y && y < max
        }
    }
    
    private static let directions = [(1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1)]
    
    let n: Int
    var pieces: [[Float]]
    
    private(set) var allMoves: [Coord] = []
    
    init(n: Int) {
        self.n = n
        self.pieces = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)
        pieces[n / 2 - 1][n / 2] = 1
        pieces[n / 2][n / 2 - 1] = 1
        pieces[n / 2 - 1][n / 2 - 1] = -1
        pieces[n / 2][n / 2] = -1
        for y in 0..<n {
            for x in 0..<n {
                allMoves.append(Coord(x, y))
            }
        }
    }
    
    func countDiff(color: Float) -> Int {
        var count = 0
        for y in 0..<n {
            for x in 0..<n {
                if pieces[x][y] == color {
                    count += 1
                } else if pieces[x][y] == -color {
                    count -= 1
                }
            }
        }
        return count
    }
    
    func legalMoves(color: Float) -> [Float] {
        var lMoves: [Coord] = []
        
        for y in 0..<n {
            for x in 0..<n {
                if pieces[x][y] == color {
                    if let newMoves = moves(for: Coord(x, y)) {
                        lMoves += newMoves
                    }
                }
            }
        }
        
        return allMoves.map({ lMoves.contains($0) ? 1 : 0 })
    }
    
    func hasLegalMoves(color: Float) -> Bool {
        for move in allMoves {
            if pieces[move.x][move.y] == color {
                if let newMoves = moves(for: move) {
                    if !newMoves.isEmpty {
                        return true
                    }
                }
            }
        }
        return false
    }
    
    func moves(for square: Coord) -> [Coord]? {
        let color = pieces[square.x][square.y]
        
        if color == 0 {
            return nil
        }
        
        var moves: [Coord] = []
        for direction in OthelloLogic.directions {
            if let move = discoverMove(origin: square, direction: direction) {
                moves.append(move)
            }
        }
        
        return moves
    }
    
    mutating func execute(move: Int, color: Float) {
        let move = allMoves[move]
        let flips = OthelloLogic.directions.map { direction -> [Coord] in
            self.flips(origin: move, direction: direction, color: color)
        }.reduce([], +)
        
        for flip in flips {
            pieces[flip.x][flip.y] = color
        }
    }
    
    func discoverMove(origin: Coord, direction: (Int, Int)) -> Coord? {
        var newMove = origin
        newMove.increment(direction)
        let color = pieces[origin.x][origin.y]
        var flips: [Coord] = []
        
        while newMove.isValid(max: n) {
            if pieces[newMove.x][newMove.y] == 0 {
                if !flips.isEmpty {
                    return newMove
                } else {
                    return nil
                }
            } else if pieces[newMove.x][newMove.y] == color {
                return nil
            } else if pieces[newMove.x][newMove.y] == -color {
                flips.append(newMove)
            }
            newMove.increment(direction)
        }
        
        return nil
    }
    
    func flips(origin: Coord, direction: (Int, Int), color: Float) -> [Coord] {
        var flips = [origin]
        var newCoord = origin
        newCoord.increment(direction)
        
        while newCoord.isValid(max: n) {
            if pieces[newCoord.x][newCoord.y] == 0 {
                return []
            }
            if pieces[newCoord.x][newCoord.y] == -color {
                flips.append(newCoord)
            }
            if pieces[newCoord.x][newCoord.y] == color && !flips.isEmpty {
                return flips
            }
            newCoord.increment(direction)
        }
        
        return []
    }
}

struct OthelloGame: Game {
    let n: Int
    
    private(set) var internalOthelloLogic: ThreadLocal<OthelloLogic>
    
    var initBoard: [[Float]]
    var boardSize: (Int, Int)
    var actionSize: Int
    
    init(n: Int) {
        self.n = n
        
        internalOthelloLogic = ThreadLocal(value: OthelloLogic(n: n))
        
        initBoard = internalOthelloLogic.inner.value.pieces
        boardSize = (n, n)
        actionSize = n * n + 1
    }
    
    func getNextState(board: [[Float]], player: Int, action: Int) -> ([[Float]], Int) {
        if action == n * n {
            return (board, -player)
        }
        internalOthelloLogic.inner.value.pieces = board
        internalOthelloLogic.inner.value.execute(move: action, color: Float(player))
        return (internalOthelloLogic.inner.value.pieces, -player)
    }
    
    func getValidMoves(board: [[Float]], player: Int) -> [Float] {
        internalOthelloLogic.inner.value.pieces = board
        var legals = internalOthelloLogic.inner.value.legalMoves(color: Float(player))
        if !legals.contains(1) {
            legals.append(1)
        } else {
            legals.append(0)
        }
        return legals
    }
    
    func getGameEnded(board: [[Float]], player: Int) -> Float {
        internalOthelloLogic.inner.value.pieces = board
        let player = Float(player)
        if internalOthelloLogic.inner.value.hasLegalMoves(color: player) {
            return 0
        }
        if internalOthelloLogic.inner.value.hasLegalMoves(color: -player) {
            return 0
        }
        if internalOthelloLogic.inner.value.countDiff(color: player) > 0 {
            return 1
        }
        return -1
    }
    
    func getCanonicalForm(board: [[Float]], player: Int) -> [[Float]] {
        return board.map({ $0 * Float(player) })
    }

    func getSymmetries(board: [[Float]], pi: [Float], valids: [Float]) -> ([[[Float]]], [[Float]], [[Float]]) {
        var piBoard = pi
        piBoard.removeLast()
        var validsBoard = valids
        validsBoard.removeLast()
        var boards: [[[Float]]] = [board]
        var pis: [[Float]] = [piBoard]
        var validss: [[Float]] = [validsBoard]

        for i in 1..<4 {
            let newB = rotate90(boards.last!, size: n, dVal: 0)
            let newPi = rotate90(pis.last!, size: n, dVal: 0)
            let newValids = rotate90(validss.last!, size: n, dVal: 0)
            boards.append(newB)
            pis.append(newPi)
            validss.append(newValids)
        }

        for i in 0..<4 {
            boards.append(flip(boards[i]))
            pis.append(flip1d(pis[i], size: n))
            validss.append(flip1d(validss[i], size: n))
        }

        pis = pis.map({ $0 + [pi.last!] })
        validss = validss.map({ $0 + [valids.last!] })

        return (boards, pis, validss)
    }
    
    func simpleRepresentation(board: [[Float]]) -> String {
        return "\(board)"
    }
}

struct OthelloModel: Layer {
    struct Input: Differentiable {
        var board: Tensor<Float>
        var valids: Tensor<Float>
    }

    struct Output: Differentiable {
        var policy: Tensor<Float>
        var value: Tensor<Float>
    }

    struct ConvBlock: Layer {
        var conv: Conv2D<Float>

        init(inputChannels: Int = 64, activation: Bool = true) {
            if activation {
                conv = Conv2D(filterShape: (3, 3, inputChannels, 64), padding: .same, activation: relu, useBias: false)
            } else {
                conv = Conv2D(filterShape: (3, 3, inputChannels, 64), padding: .same, useBias: false)
            }
        }

        func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            return conv(input)
        }
    }

    struct ResidualBlock: Layer {
        var conv1: ConvBlock = .init()
        var conv2: ConvBlock = .init(activation: false)

        init() {}

        func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            return relu(input + conv2(conv1(input)))
        }
    }

    struct DenseBlock: Layer {
        var weight: Tensor<Float>
        var dropout: Dropout<Float>

        init(inputs: Int, outputs: Int, dropoutProb: Double) {
            weight = Tensor(randomUniform: [inputs, outputs])
            dropout = Dropout(probability: dropoutProb)
        }

        func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            return dropout(relu(matmul(input, weight)))
        }
    }

    @noDerivative private let r, c: Int

    var cb: ConvBlock
    var res: [ResidualBlock] = []
    @noDerivative let flattener: Flatten<Float> = Flatten()
    var db1: DenseBlock
    var db2: DenseBlock
    var pi: Dense<Float>
    var v: Dense<Float>

    init(game: OthelloGame, dropout: Double) {
        (self.r, self.c) = (game.n, game.n)
        cb = ConvBlock(inputChannels: 1)
        for _ in 1...10 {
            res.append(ResidualBlock())
        }
        db1 = DenseBlock(inputs: r * c * 64, outputs: 1024, dropoutProb: dropout)
        db2 = DenseBlock(inputs: 1024, outputs: 512, dropoutProb: dropout)
        pi = Dense(inputSize: 512, outputSize: game.actionSize)
        v = Dense(inputSize: 512, outputSize: 1, activation: tanh)
    }

    func callAsFunction(_ input: Input) -> Output {
        var convOutput = cb(input.board.reshaped(to: [input.board.shape[0], r, c, 1]))
        for i in withoutDerivative(at: 0..<res.count) {
            convOutput = res[i](convOutput)
        }
        let dense = db2(db1(flattener(convOutput)))
        let policy = pi(dense)
        let policyFixed = softmax(policy - ((Tensor<Float>(1.0) - input.valids.reshaped(to: policy.shape)) * Tensor<Float>(1000.0)))
        return .init(policy: policyFixed, value: v(dense))
    }
}

struct OthelloNNet: NNet {
    static let pyQueue = DispatchQueue(label: "py")
    
    struct PyInterface {
        var pyModel: PythonObject! = nil
        
        init() {
            OthelloNNet.pyQueue.sync {
                sys.path.append(OTHELLO_PATH)
                let pyTrainer = Python.import("othello_trainer")
                pyModel = pyTrainer.PyOthelloModel()
            }
        }
    }
    
    let pyInterface: PyInterface = PyInterface()
    var model: OthelloModel
    let game: OthelloGame
    
    var dropout = 0.3
    var epochs = 35
    var batchSize = 16384
    
    init(game: OthelloGame) {
        self.game = game
        self.model = OthelloModel(game: game, dropout: 0.3)
        OthelloNNet.pyQueue.sync {
            importPyWeights()
        }
    }
    
    mutating func train() {
        if !Thread.isMainThread {
            fatalError()
        }
        pyInterface.pyModel.train_model(epochs: epochs, batch_size: batchSize)
        importPyWeights()
    }

    mutating func storeEpisode(boards: [[[Float]]], pis: [[Float]], vs: [[Float]], valids: [[Float]]) {
        let boards = toTensor(boards).makeNumpyArray()
        let pis = toTensor(pis).makeNumpyArray()
        let vs = toTensor(vs).makeNumpyArray()
        let valids = toTensor(valids).makeNumpyArray()
        pyInterface.pyModel.add_data([boards, valids, pis, vs])
    }
    
    mutating func importPyWeights() {
        let pyWeights = pyInterface.pyModel.get_swift_weights()
        var kpIdx = 0
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            let swiftShape = model[keyPath: kp].shapeTensor.scalars
            if swiftShape.isEmpty {
                continue
            }
            let pyShape = [Int32](Python.list(pyWeights[kpIdx].shape))!
            if swiftShape != pyShape {
                fatalError()
            }
            model[keyPath: kp] = Tensor<Float>(numpy: pyWeights[kpIdx].astype("float32"))!
            kpIdx += 1
        }

        let randomInputBoard = Tensor<Float>(randomUniform: [1, 8, 8])
        let randomInputLegals = Tensor<Float>(ones: [1, 65])
        let swiftPred = model(.init(board: randomInputBoard, valids: randomInputLegals))
        let swiftPolicy = Tensor(swiftPred.policy[0].scalars)
        let kerasPred = pyInterface.pyModel.model.predict([randomInputBoard.makeNumpyArray(), randomInputLegals.makeNumpyArray()])
        let kerasPolicy = Tensor<Float>(numpy: kerasPred[0])!
        let msePolicy = pow(swiftPolicy - kerasPolicy, Tensor(2.0)).mean()
        print("KERAS->S4TF CONVERSION MSE: \(msePolicy)")
    }
    
    func predict(board: [[Float]], valids: [Float]) -> ([Float], Float) {
        let board = toTensor(board)
        let valids = toTensor(valids).reshaped(to: [1, game.actionSize, 1])
        let prediction = model(.init(board: board.expandingShape(at: 0), valids: valids))
        return (prediction.policy[0].scalars, prediction.value[0][0].scalar!)
    }
    
    func save(checkpoint: String) {
        pyInterface.pyModel.model.save_weights("\(OTHELLO_PATH)/\(checkpoint)")
    }
    
    mutating func load(checkpoint: String) {
        pyInterface.pyModel.model.load_weights("\(OTHELLO_PATH)/\(checkpoint)")
        importPyWeights()
    }
}
