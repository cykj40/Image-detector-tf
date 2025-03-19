/**
 * Shape Detector using Computer Vision techniques
 * Pure JavaScript implementation without external dependencies
 */

// Configuration for the detector
const CONFIG = {
    // Edge detection
    cannyThreshold1: 10,
    cannyThreshold2: 50,

    // Circle detection
    circleMinRadius: 10,
    circleMaxRadius: 100,
    circularity: 0.6,

    // Triangle detection
    triangleMinArea: 30,
    triangleCorners: 3,
    triangleTolerance: 0.25,

    // General
    contourMinArea: 20,
    debug: true,
    thresholdValue: 5
};

/**
 * Detects shapes in an image
 * @param {HTMLCanvasElement} canvas - The canvas element containing the drawing
 * @returns {Object} Detection result with shape, confidence, and debug info
 */
function detectShape(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Get image data
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    // Debug message
    console.log("Processing canvas of size: " + width + "x" + height);

    // Create grayscale image
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i += 4) {
        // Convert RGB to grayscale using luminance formula
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const a = data[i + 3]; // Consider alpha channel for transparency
        // If pixel is not transparent at all, boost its intensity
        const boost = a > 200 ? 1.5 : 1.0;
        const luminance = (0.299 * r + 0.587 * g + 0.114 * b) * boost;
        gray[i / 4] = Math.min(255, luminance);
    }

    // Threshold the image to create a binary image
    const binary = new Uint8Array(width * height);
    const threshold = CONFIG.thresholdValue;
    for (let i = 0; i < gray.length; i++) {
        binary[i] = gray[i] > threshold ? 255 : 0;
    }

    // Find contours in the binary image
    const contours = findContours(binary, width, height);
    console.log("Found " + contours.length + " contours");

    // Debug visualization
    const debugCanvas = document.createElement('canvas');
    debugCanvas.width = width;
    debugCanvas.height = height;
    const debugCtx = debugCanvas.getContext('2d');
    debugCtx.putImageData(imageData, 0, 0);

    // Draw all contours for debugging
    if (CONFIG.debug) {
        debugCtx.strokeStyle = 'blue';
        debugCtx.lineWidth = 1;

        contours.forEach(contour => {
            debugCtx.beginPath();
            for (let i = 0; i < contour.length; i++) {
                const [x, y] = contour[i];
                if (i === 0) {
                    debugCtx.moveTo(x, y);
                } else {
                    debugCtx.lineTo(x, y);
                }
            }
            debugCtx.closePath();
            debugCtx.stroke();
        });
    }

    // Filter contours by area
    const validContours = contours.filter(contour =>
        calculateContourArea(contour) > CONFIG.contourMinArea
    );
    console.log("Found " + validContours.length + " valid contours");

    if (validContours.length === 0) {
        return {
            shape: 'unknown',
            confidence: 0,
            debugImage: debugCanvas.toDataURL(),
            message: 'No valid contours found'
        };
    }

    // Find the largest contour
    const mainContour = validContours.reduce((prev, current) =>
        calculateContourArea(current) > calculateContourArea(prev) ? current : prev
    );

    // Draw the main contour
    if (CONFIG.debug) {
        debugCtx.strokeStyle = 'lime';
        debugCtx.lineWidth = 2;
        debugCtx.beginPath();
        for (let i = 0; i < mainContour.length; i++) {
            const [x, y] = mainContour[i];
            if (i === 0) {
                debugCtx.moveTo(x, y);
            } else {
                debugCtx.lineTo(x, y);
            }
        }
        debugCtx.closePath();
        debugCtx.stroke();
    }

    // Calculate shape features
    const area = calculateContourArea(mainContour);
    const perimeter = calculateContourPerimeter(mainContour);
    const circularity = calculateCircularity(area, perimeter);
    const convexHull = calculateConvexHull(mainContour);
    const convexArea = calculateContourArea(convexHull);
    const solidity = area / convexArea;

    // Calculate centroid
    const centroid = calculateCentroid(mainContour);

    // Get the number of corners/vertices (approximated)
    const epsilon = CONFIG.triangleTolerance * perimeter;
    const approximatedPolygon = approxPolyDP(mainContour, epsilon);
    const numCorners = approximatedPolygon.length;

    // Detected shape metrics
    let isCircle = false;
    let isTriangle = false;
    let circleConfidence = 0;
    let triangleConfidence = 0;

    // Circle detection
    if (circularity > CONFIG.circularity) {
        isCircle = true;
        circleConfidence = Math.min(1, (circularity - CONFIG.circularity) / (1 - CONFIG.circularity));
    }

    // Triangle detection
    if (numCorners === 3 && solidity > 0.85) {
        isTriangle = true;
        triangleConfidence = Math.min(1, solidity);

        // Draw the approximated triangle
        if (CONFIG.debug) {
            debugCtx.strokeStyle = 'red';
            debugCtx.lineWidth = 2;
            debugCtx.beginPath();
            for (let i = 0; i < approximatedPolygon.length; i++) {
                const [x, y] = approximatedPolygon[i];
                if (i === 0) {
                    debugCtx.moveTo(x, y);
                } else {
                    debugCtx.lineTo(x, y);
                }
            }
            debugCtx.closePath();
            debugCtx.stroke();
        }
    }

    // Draw the centroid
    if (CONFIG.debug) {
        debugCtx.fillStyle = 'blue';
        debugCtx.beginPath();
        debugCtx.arc(centroid.x, centroid.y, 4, 0, 2 * Math.PI);
        debugCtx.fill();
    }

    // Determine the shape
    let detectedShape = 'unknown';
    let confidence = 0;

    // If we have approximated polygon
    if (approximatedPolygon && approximatedPolygon.length > 0) {
        // Draw the approximated polygon for debugging
        if (CONFIG.debug) {
            debugCtx.strokeStyle = 'purple';
            debugCtx.lineWidth = 2;
            debugCtx.beginPath();
            for (let i = 0; i < approximatedPolygon.length; i++) {
                const [x, y] = approximatedPolygon[i];
                if (i === 0) {
                    debugCtx.moveTo(x, y);
                } else {
                    debugCtx.lineTo(x, y);
                }
            }
            debugCtx.closePath();
            debugCtx.stroke();
        }
    }

    if (isCircle && !isTriangle) {
        detectedShape = 'circle';
        confidence = circleConfidence;
    } else if (isTriangle && !isCircle) {
        detectedShape = 'triangle';
        confidence = triangleConfidence;
    } else if (isCircle && isTriangle) {
        if (circleConfidence > triangleConfidence) {
            detectedShape = 'circle';
            confidence = circleConfidence - triangleConfidence;
        } else {
            detectedShape = 'triangle';
            confidence = triangleConfidence - circleConfidence;
        }
    } else {
        // Not clearly a circle or triangle, make a best guess based on vertex count
        if (numCorners <= 4) {
            detectedShape = 'triangle';
            confidence = 0.5; // More confident guess
        } else {
            detectedShape = 'circle';
            confidence = 0.5; // More confident guess
        }
    }

    // Log details for debugging
    console.log("Shape metrics:", {
        area,
        perimeter,
        circularity,
        solidity,
        numCorners,
        isCircle,
        isTriangle,
        circleConfidence,
        triangleConfidence
    });

    return {
        shape: detectedShape,
        confidence: confidence,
        metrics: {
            area,
            perimeter,
            circularity,
            solidity,
            numCorners
        },
        debugImage: debugCanvas.toDataURL(),
        message: `Detected ${detectedShape} with confidence ${(confidence * 100).toFixed(1)}%`
    };
}

/**
 * Finds contours in a binary image
 * @param {Uint8Array} binary - Binary image data (0 or 255)
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @returns {Array} Array of contours, each contour is an array of [x,y] points
 */
function findContours(binary, width, height) {
    // Apply a preprocessing step to enhance faint strokes
    const enhanced = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        // Boost the signal - any non-zero value becomes full intensity
        enhanced[i] = binary[i] > 0 ? 255 : 0;
    }

    // Apply a simple blur to connect broken lines and reduce noise
    const blurred = new Uint8Array(enhanced.length);
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            // Simple 3x3 box blur
            let sum = 0;
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    sum += enhanced[(y + dy) * width + (x + dx)];
                }
            }

            // Lower threshold to catch more features
            blurred[idx] = sum > (255 * 1) ? 255 : 0; // Was 255 * 5
        }
    }

    // Simple contour finding algorithm on the blurred image
    const visited = new Uint8Array(blurred.length);
    const contours = [];

    // Find starting points for contours (pixels with value 255)
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            if (blurred[idx] === 255 && visited[idx] === 0) {
                // Found a new contour starting point
                const contour = traceContour(blurred, visited, width, height, x, y);
                if (contour.length > 2) {  // Was 3, reduce minimum size
                    contours.push(contour);
                }
            }
        }
    }

    return contours;
}

/**
 * Traces a contour starting at (startX, startY)
 * @param {Uint8Array} binary - Binary image data
 * @param {Uint8Array} visited - Visited pixels map
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @param {number} startX - Starting X coordinate
 * @param {number} startY - Starting Y coordinate
 * @returns {Array} Contour as array of [x,y] points
 */
function traceContour(binary, visited, width, height, startX, startY) {
    const contour = [];
    const stack = [[startX, startY]];
    const directions = [
        [-1, -1], [0, -1], [1, -1],
        [-1, 0], [1, 0],
        [-1, 1], [0, 1], [1, 1]
    ];

    // Mark the first point to avoid infinite loop
    visited[startY * width + startX] = 1;
    contour.push([startX, startY]);

    while (stack.length > 0) {
        const [x, y] = stack.pop();

        // Try all 8 directions
        let foundDirection = false;
        for (const [dx, dy] of directions) {
            const nx = x + dx;
            const ny = y + dy;

            // Check bounds
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                continue;
            }

            const idx = ny * width + nx;

            // If this pixel is part of the shape and not visited
            if (binary[idx] === 255 && visited[idx] === 0) {
                visited[idx] = 1;
                contour.push([nx, ny]);
                stack.push([nx, ny]);
                foundDirection = true;
                break;  // Only follow one direction at a time
            }
        }

        // If we couldn't find any direction to go, we might be at the end of a branch
        if (!foundDirection && stack.length === 0 && contour.length > 1) {
            // Try to find the next closest unvisited pixel
            let found = false;
            for (let r = 2; r <= 3 && !found; r++) { // Search in increasing radius
                for (let dy = -r; dy <= r && !found; dy++) {
                    for (let dx = -r; dx <= r && !found; dx++) {
                        if (dx === 0 && dy === 0) continue;

                        const nx = x + dx;
                        const ny = y + dy;

                        // Check bounds
                        if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                            continue;
                        }

                        const idx = ny * width + nx;

                        // If this pixel is part of the shape and not visited
                        if (binary[idx] === 255 && visited[idx] === 0) {
                            visited[idx] = 1;
                            contour.push([nx, ny]);
                            stack.push([nx, ny]);
                            found = true;
                        }
                    }
                }
            }
        }
    }

    return contour;
}

/**
 * Calculates the area of a contour using the Shoelace formula
 * @param {Array} contour - Array of [x,y] points
 * @returns {number} Area of the contour
 */
function calculateContourArea(contour) {
    let area = 0;
    const n = contour.length;

    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        area += contour[i][0] * contour[j][1];
        area -= contour[j][0] * contour[i][1];
    }

    return Math.abs(area / 2);
}

/**
 * Calculates the perimeter of a contour
 * @param {Array} contour - Array of [x,y] points
 * @returns {number} Perimeter of the contour
 */
function calculateContourPerimeter(contour) {
    let perimeter = 0;
    const n = contour.length;

    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        const dx = contour[j][0] - contour[i][0];
        const dy = contour[j][1] - contour[i][1];
        perimeter += Math.sqrt(dx * dx + dy * dy);
    }

    return perimeter;
}

/**
 * Calculates the circularity of a contour (1 for perfect circle)
 * @param {number} area - Area of the contour
 * @param {number} perimeter - Perimeter of the contour
 * @returns {number} Circularity measure (0-1)
 */
function calculateCircularity(area, perimeter) {
    return (4 * Math.PI * area) / (perimeter * perimeter);
}

/**
 * Calculates the centroid of a contour
 * @param {Array} contour - Array of [x,y] points
 * @returns {Object} Centroid coordinates {x, y}
 */
function calculateCentroid(contour) {
    let sumX = 0;
    let sumY = 0;

    for (let i = 0; i < contour.length; i++) {
        sumX += contour[i][0];
        sumY += contour[i][1];
    }

    return {
        x: sumX / contour.length,
        y: sumY / contour.length
    };
}

/**
 * Calculates the convex hull of a contour using Graham scan
 * @param {Array} contour - Array of [x,y] points
 * @returns {Array} Convex hull as array of [x,y] points
 */
function calculateConvexHull(contour) {
    // Find the point with the lowest y-coordinate (and leftmost if tied)
    let start = 0;
    for (let i = 1; i < contour.length; i++) {
        if (contour[i][1] < contour[start][1] ||
            (contour[i][1] === contour[start][1] && contour[i][0] < contour[start][0])) {
            start = i;
        }
    }

    // Swap the start point to the beginning
    [contour[0], contour[start]] = [contour[start], contour[0]];

    // Sort points by polar angle with respect to the start point
    const startPoint = contour[0];
    const sortedPoints = contour.slice(1).sort((a, b) => {
        const angleA = Math.atan2(a[1] - startPoint[1], a[0] - startPoint[0]);
        const angleB = Math.atan2(b[1] - startPoint[1], b[0] - startPoint[0]);
        return angleA - angleB;
    });

    // Graham scan algorithm
    const hull = [startPoint, sortedPoints[0]];

    for (let i = 1; i < sortedPoints.length; i++) {
        while (hull.length > 1 && !isCounterClockwise(
            hull[hull.length - 2],
            hull[hull.length - 1],
            sortedPoints[i])) {
            hull.pop();
        }
        hull.push(sortedPoints[i]);
    }

    return hull;
}

/**
 * Checks if three points make a counter-clockwise turn
 * @param {Array} p1 - First point [x,y]
 * @param {Array} p2 - Second point [x,y]
 * @param {Array} p3 - Third point [x,y]
 * @returns {boolean} True if counter-clockwise
 */
function isCounterClockwise(p1, p2, p3) {
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) > 0;
}

/**
 * Approximates a polygon with fewer points (Douglas-Peucker algorithm)
 * @param {Array} contour - Array of [x,y] points
 * @param {number} epsilon - Maximum distance from point to line
 * @returns {Array} Simplified contour
 */
function approxPolyDP(contour, epsilon) {
    if (contour.length <= 2) {
        return contour;
    }

    // Find the point with the maximum distance from the line
    let maxDist = 0;
    let index = 0;

    const firstPoint = contour[0];
    const lastPoint = contour[contour.length - 1];

    for (let i = 1; i < contour.length - 1; i++) {
        const dist = perpendicularDistance(contour[i], firstPoint, lastPoint);
        if (dist > maxDist) {
            maxDist = dist;
            index = i;
        }
    }

    // If max distance is greater than epsilon, recursively simplify
    if (maxDist > epsilon) {
        // Recursive call
        const firstHalf = approxPolyDP(contour.slice(0, index + 1), epsilon);
        const secondHalf = approxPolyDP(contour.slice(index), epsilon);

        // Combine results, removing the duplicate point
        return [...firstHalf.slice(0, -1), ...secondHalf];
    } else {
        // All points are within epsilon, so use just the endpoints
        return [firstPoint, lastPoint];
    }
}

/**
 * Calculates the perpendicular distance from point to line
 * @param {Array} point - Point [x,y]
 * @param {Array} lineStart - Line start point [x,y]
 * @param {Array} lineEnd - Line end point [x,y]
 * @returns {number} Distance from point to line
 */
function perpendicularDistance(point, lineStart, lineEnd) {
    const [x, y] = point;
    const [x1, y1] = lineStart;
    const [x2, y2] = lineEnd;

    const area = Math.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1);
    const length = Math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2);

    return area / length;
}

// Export the shape detector
export { detectShape }; 