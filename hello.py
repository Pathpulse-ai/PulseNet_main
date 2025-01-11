print("Hello.py")

import Foundation


func calculateDistance(userLocation: (Double, Double), driverLocation: (Double, Double)) -> Double {
    let xDiff = userLocation.0 - driverLocation.0
    let yDiff = userLocation.1 - driverLocation.1
    return sqrt(xDiff * xDiff + yDiff * yDiff)
}


func top3ClosestDrivers(
    drivers: [(name: String, location: (Double, Double), rating: Double)],
    userLocation: (Double, Double)
) -> [String] {
    // Calculate distance for each driver and sort by conditions
    let sortedDrivers = drivers.sorted {
        let distance1 = calculateDistance(userLocation: userLocation, driverLocation: $0.location)
        let distance2 = calculateDistance(userLocation: userLocation, driverLocation: $1.location)
        
        if distance1 == distance2 {
            if $0.rating == $1.rating {
                return $0.name < $1.name // Lexicographic order
            }
            return $0.rating > $1.rating // Higher rating
        }
        return distance1 < distance2 // Closer distance
    }
    
    
    return sortedDrivers.prefix(3).map { $0.name }
}


let drivers = [
    (name: "Alice", location: (2.0, 3.0), rating: 4.5),
    (name: "Bob", location: (5.0, 1.0), rating: 4.8),
    (name: "Charlie", location: (1.0, 2.0), rating: 4.5),
    (name: "David", location: (3.0, 4.0), rating: 4.0),
    (name: "Eve", location: (2.5, 3.5), rating: 4.9)
]
let userLocation = (2.0, 3.0)

let closestDrivers = top3ClosestDrivers(drivers: drivers, userLocation: userLocation)
print(closestDrivers)
// Output: ["Alice", "Eve", "Charlie"]
