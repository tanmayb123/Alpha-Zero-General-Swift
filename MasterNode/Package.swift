// swift-tools-version:5.0
import PackageDescription

let package = Package(
    name: "MasterNode",
    platforms: [
        .macOS(.v10_14),
    ],
    products: [
        .library(name: "MasterNode", targets: ["App"]),
    ],
    dependencies: [
        .package(url: "https://github.com/vapor/vapor.git", from: "3.0.0"),
        .package(url: "https://github.com/vapor/fluent-sqlite.git", from: "3.0.0"),
        .package(url: "https://github.com/Alamofire/Alamofire.git", .upToNextMajor(from: "5.1.0"))
    ],
    targets: [
        .target(name: "App", dependencies: ["Alamofire", "FluentSQLite", "Vapor"]),
        .target(name: "Run", dependencies: ["App"]),
    ]
)

