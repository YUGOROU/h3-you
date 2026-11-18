import Foundation

/// Preserves user/lifecycle command order across asynchronous executor hops.
@MainActor
final class SerialCommands {
    private var last: Task<Void, Never>?

    @discardableResult
    func enqueue(_ operation: @escaping @Sendable () async -> Void) -> Task<Void, Never> {
        let previous = last
        let command = Task {
            await previous?.value
            await operation()
        }
        last = command
        return command
    }
}
