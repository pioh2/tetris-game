const canvas = document.getElementById('tetris-grid');
const context = canvas.getContext('2d');
const scoreDisplay = document.getElementById('score-display');
const gameOverMessage = document.getElementById('game-over-message');
const leftButton = document.getElementById('left-button');
const rightButton = document.getElementById('right-button');
const downButton = document.getElementById('down-button');
const rotateButton = document.getElementById('rotate-button');
const speedInput = document.getElementById('speed-input');

const grid = createGrid(20, 10);
let currentBlock = createBlock();
let score = 0;
let gameOver = false;
let paused = false;
let dropInterval = 1000;
let dropCounter = 0;
let lastTime = 0;

function createGrid(rows, cols) {
    const grid = [];
    for (let row = 0; row < rows; row++) {
        grid.push(new Array(cols).fill(0));
    }
    return grid;
}

function createBlock() {
    const blocks = [
        [[1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[0, 1, 0], [1, 1, 1]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0]]
    ];
    const block = blocks[Math.floor(Math.random() * blocks.length)];
    return { shape: block, row: 0, col: Math.floor(grid[0].length / 2) - 1 };
}

function drawGrid() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (let row = 0; row < grid.length; row++) {
        for (let col = 0; col < grid[row].length; col++) {
            if (grid[row][col]) {
                context.fillStyle = 'white';
                context.fillRect(col * 30, row * 30, 30, 30);
            }
        }
    }
}

function drawBlock(block) {
    context.fillStyle = 'white';
    for (let row = 0; row < block.shape.length; row++) {
        for (let col = 0; col < block.shape[row].length; col++) {
            if (block.shape[row][col]) {
                context.fillRect((block.col + col) * 30, (block.row + row) * 30, 30, 30);
            }
        }
    }
}

function moveBlockDown() {
    if (!paused && !gameOver) {
        currentBlock.row++;
        if (collision()) {
            currentBlock.row--;
            mergeBlock();
            currentBlock = createBlock();
            if (collision()) {
                gameOver = true;
                gameOverMessage.style.display = 'block';
            }
        }
        drawGrid();
        drawBlock(currentBlock);
    }
}

function collision() {
    for (let row = 0; row < currentBlock.shape.length; row++) {
        for (let col = 0; col < currentBlock.shape[row].length; col++) {
            if (currentBlock.shape[row][col] &&
                (grid[currentBlock.row + row] &&
                grid[currentBlock.row + row][currentBlock.col + col]) !== 0) {
                return true;
            }
        }
    }
    return false;
}

function mergeBlock() {
    for (let row = 0; row < currentBlock.shape.length; row++) {
        for (let col = 0; col < currentBlock.shape[row].length; col++) {
            if (currentBlock.shape[row][col]) {
                grid[currentBlock.row + row][currentBlock.col + col] = 1;
            }
        }
    }
    clearLines();
}

function clearLines() {
    for (let row = grid.length - 1; row >= 0; row--) {
        if (grid[row].every(cell => cell !== 0)) {
            grid.splice(row, 1);
            grid.unshift(new Array(grid[0].length).fill(0));
            score += 10;
            scoreDisplay.textContent = `Score: ${score}`;
        }
    }
}

function handleInput(event) {
    if (!paused && !gameOver) {
        switch (event.key) {
            case 'ArrowLeft':
                currentBlock.col--;
                if (collision()) {
                    currentBlock.col++;
                }
                break;
            case 'ArrowRight':
                currentBlock.col++;
                if (collision()) {
                    currentBlock.col--;
                }
                break;
            case 'ArrowDown':
                moveBlockDown();
                break;
            case 'ArrowUp':
                rotateBlock();
                break;
        }
        drawGrid();
        drawBlock(currentBlock);
    }
}

function rotateBlock() {
    const rotatedShape = currentBlock.shape[0].map((_, index) =>
        currentBlock.shape.map(row => row[index]).reverse()
    );
    const originalCol = currentBlock.col;
    currentBlock.col -= Math.floor((rotatedShape[0].length - currentBlock.shape[0].length) / 2);
    currentBlock.shape = rotatedShape;
    if (collision()) {
        currentBlock.shape = rotatedShape[0].map((_, index) =>
            rotatedShape.map(row => row[index]).reverse()
        );
        currentBlock.col = originalCol;
    }
}

function togglePause() {
    paused = !paused;
}

function update(time = 0) {
    const deltaTime = time - lastTime;
    lastTime = time;
    dropCounter += deltaTime;
    if (dropCounter > dropInterval) {
        moveBlockDown();
        dropCounter = 0;
    }
    drawGrid();
    drawBlock(currentBlock);
    if (!gameOver) {
        requestAnimationFrame(update);
    }
}

function handleTouchInput(event) {
    if (!paused && !gameOver) {
        switch (event.target.id) {
            case 'left-button':
                currentBlock.col--;
                if (collision()) {
                    currentBlock.col++;
                }
                break;
            case 'right-button':
                currentBlock.col++;
                if (collision()) {
                    currentBlock.col--;
                }
                break;
            case 'down-button':
                moveBlockDown();
                break;
            case 'rotate-button':
                rotateBlock();
                break;
        }
        drawGrid();
        drawBlock(currentBlock);
    }
}

function handleSpeedInput(event) {
    dropInterval = 1000 - event.target.value;
}

document.addEventListener('keydown', handleInput);
document.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
        togglePause();
    }
});

leftButton.addEventListener('click', handleTouchInput);
rightButton.addEventListener('click', handleTouchInput);
downButton.addEventListener('click', handleTouchInput);
rotateButton.addEventListener('click', handleTouchInput);
speedInput.addEventListener('input', handleSpeedInput);

update();
