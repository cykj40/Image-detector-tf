const fs = require('fs');
const path = require('path');

console.log('Starting model file copy process...');
console.log('Current directory:', __dirname);

// Define all possible model directories
const directories = [
    path.join(__dirname, 'dist', 'model'),        // Standard dist/model
    path.join(__dirname, 'dist'),                 // Direct in dist
    path.join(__dirname, 'model')                 // Root model directory
];

// Create directories if they don't exist
directories.forEach(dir => {
    if (!fs.existsSync(dir)) {
        console.log(`Creating directory: ${dir}`);
        fs.mkdirSync(dir, { recursive: true });
    }
});

// Source model directory
const sourceModelDir = path.join(__dirname, 'public', 'model');
console.log('Source model directory:', sourceModelDir);

// Check if source directory exists
if (!fs.existsSync(sourceModelDir)) {
    console.error('ERROR: Source model directory does not exist!');
    process.exit(1);
}

// List all files in the source directory
console.log('Files in source directory:');
const sourceFiles = fs.readdirSync(sourceModelDir);
sourceFiles.forEach(file => console.log(`- ${file} (${fs.statSync(path.join(sourceModelDir, file)).size} bytes)`));

// Files to copy
const files = ['model.json', 'weights.bin'];

// Copy model files to all possible locations
directories.forEach(destDir => {
    console.log(`\nCopying files to: ${destDir}`);

    files.forEach(file => {
        const sourcePath = path.join(sourceModelDir, file);
        const destPath = path.join(destDir, file);

        if (fs.existsSync(sourcePath)) {
            fs.copyFileSync(sourcePath, destPath);
            const sourceSize = fs.statSync(sourcePath).size;
            const destSize = fs.statSync(destPath).size;

            if (sourceSize === destSize) {
                console.log(`✓ Successfully copied ${file} (${sourceSize} bytes) to ${destDir}`);
            } else {
                console.error(`⚠ Warning: Size mismatch in ${file}. Source: ${sourceSize} bytes, Dest: ${destSize} bytes`);
            }
        } else {
            console.error(`❌ ERROR: ${file} not found in public/model/`);
        }
    });

    // List files in destination directory
    console.log(`Files in ${destDir}:`);
    if (fs.existsSync(destDir)) {
        const destFiles = fs.readdirSync(destDir);
        destFiles.forEach(file => console.log(`- ${file} (${fs.statSync(path.join(destDir, file)).size} bytes)`));
    }
});

console.log('\nModel file copy process complete!'); 