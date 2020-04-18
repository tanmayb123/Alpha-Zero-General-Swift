import Foundation
import Alamofire
import Vapor

/*

Master<->Worker Architecture

1. Master has list of worker IPs.
3. Master calls `/getNetwork` on all workers with URL to compressed weights.
4. Worker downloads weights from URL given by Master.
5. Master calls `/generate` on all workers with Run UUID.
6. Worker calls `/newdata` with episode data & associated Run UUID.
7. Master calls `/stop` on all workers once episodes are complete.

*/

@discardableResult
func shell(_ args: String...) -> Int32 {
    let task = Process()
    task.launchPath = "/usr/bin/env"
    task.arguments = args
    task.launch()
    task.waitUntilExit()
    return task.terminationStatus
}

var game = OthelloGame(n: 8)
var nnet = OthelloNNet(game: game)
var worker = WorkerNode(game: game, nnet: nnet)

let THREADS = 5
let MASTER_DATA = "http://35.239.213.124:4444/"
let MASTER_MODEL = "http://35.239.213.124:3333/"

class WorkerNode<Task: Game, Network: NNet> where Network.Task == Task {
    var executor: EpisodeExecutor<Task, Network>
    var shouldGenerate = false

    init(game: Task, nnet: Network) {
        executor = EpisodeExecutor(game: game, nnet: nnet)
    }

    func downloadWeights(id: String) {
        let weightURL = MASTER_MODEL + "master\(id).h5"
        let savePath = "/Users/tanmaybakshi/othello/master\(id).h5"
        print("`wget` terminated with code: \(shell("wget", weightURL, "-O", savePath))")
    }

    func loadNewWeights(id: String) {
        executor.nnet.load(checkpoint: "master\(id).h5")
        print("Loaded new checkpoint with id \(id)")
    }

    func generate(id: String) {
        shouldGenerate = true
        let semaphore = DispatchSemaphore(value: THREADS)
        while shouldGenerate {
            semaphore.wait()
            DispatchQueue.global().async { [self] in
                defer { semaphore.signal() }
                guard shouldGenerate else {
                    return
                }
                let data = executor.executeEpisode()
                let postParams: [String: Any] = [
                    "id": id,
                    "boards": data.boards,
                    "pis": data.pis,
                    "vs": data.vs,
                    "valids": data.valids
                ]
                guard shouldGenerate else {
                    return
                }
                _ = AF.request(MASTER_DATA + "newdata", method: .post, parameters: postParams, encoding: JSONEncoding.default)
                print("Episode generated & shipped")
            }
        }
    }
}

public func routes(_ router: Router) throws {
    router.get("init") { req -> String in
        print("Worker got init from master.")
        return ""
    }

    router.get("getNetwork") { req -> String in
        print("Worker got getNetwork from master.")
        let modelId = req.query[String.self, at: "id"]!
        worker.downloadWeights(id: modelId)
        worker.loadNewWeights(id: modelId)
        return ""
    }

    router.get("generate") { req -> String in
        print("Worker got generate from master.")
        let runId = req.query[String.self, at: "id"]!
        worker.generate(id: runId)
        return ""
    }

    router.get("stop") { req -> String in
        print("Worker got stop from master.")
        if worker.shouldGenerate {
            print("CORRUPTED STATE! Told to stop when worker wasn't generating.")
        }
        worker.shouldGenerate = false
        return ""
    }
}
