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

var coach: DistributedCoach<OthelloGame, OthelloNNet>!

public func routes(_ router: Router) throws {
    router.get("learn") { req -> String in
        print("Learning!")
        DispatchQueue.global().async {
            var game = OthelloGame(n: 8)
            var nnet = OthelloNNet(game: game)
            coach = DistributedCoach(game: game, nnet: nnet)
            coach.learn()
        }
        return ""
    }

    router.post("newdata") { req -> String in
        let data = try! req.content.decode(DistributedCoach<OthelloGame, OthelloNNet>.TrainExamples.self)
        data.map { data in
            coach.gotEpisode(data: data)
        }
        return ""
    }
}
