import Foundation

/// Waits for consecutive matching observations, rather than treating a preferred
/// route request as an immediately effective route. No hardware dependencies.
enum AudioRouteSettlement {
    static func wait(
        maximumSamples: Int = 31,
        requiredConsecutiveSamples: Int = 3,
        sample: @Sendable () async throws -> Bool,
        pause: @Sendable () async throws -> Void = { try await Task.sleep(for: .milliseconds(100)) }
    ) async throws -> Bool {
        precondition(maximumSamples > 0 && requiredConsecutiveSamples > 0)
        var consecutive = 0
        for index in 0..<maximumSamples {
            try Task.checkCancellation()
            let matches = try await sample()
            try Task.checkCancellation()
            consecutive = matches ? consecutive + 1 : 0
            if consecutive >= requiredConsecutiveSamples { return true }
            if index + 1 < maximumSamples { try await pause() }
        }
        return false
    }
}
