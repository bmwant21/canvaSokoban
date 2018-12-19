var ItemsEnum = {
  WALL: -1
};

man = new Image();
block = new Image();
box = new Image();
boxCompleted = new Image();
cherry = new Image();
moves = 0;
cherriesLeft = 0;

var field, posi, posj,
  endi, endj,
  totalMoves, path;

$.get('/load_data', function(data) {
  console.log(data);
  field = data.field;
  posj = data.player.x;
  posi = data.player.y;
  endj = data.end.x;
  endi = data.end.y;
  path = data.path;
  totalMoves = data.moves;
  draw();
  drawScores();
  drawStats();
  // drawAllField();
});

var cherries = [
  [3, 3],
  // [12, 2],
  // [13, 12]
];

// shim layer with setTimeout fallback
window.requestAnimFrame = (function() {
  return  window.requestAnimationFrame   ||
    window.webkitRequestAnimationFrame ||
    window.mozRequestAnimationFrame    ||
    window.oRequestAnimationFrame      ||
    window.msRequestAnimationFrame     ||
    function( callback ){
      window.setTimeout(callback, 1000 / 60);
    };
})();

function isCherry(i, j) {
  for (var index = 0; index < cherries.length; index++) {
    if (cherries[index][0] == i && cherries[index][1] == j) {
      return true;
    }
  }
  return false;
}

function drawElem(picture, what) {
  var canvas = document.getElementById('fieldb');
  var ctx = canvas.getContext('2d');
  for(var i = 0; i < field.length; i++) {
    for(var j = 0; j < field[i].length; j++) {
      var coordx = 20*j;
      var coordy = 20*i;

      if(field[i][j] === what) {
        ctx.drawImage(picture, coordx, coordy, 20, 20);
      }
    }
  }
}

function drawPlayer(posI, posJ) {
  var canvas = document.getElementById('fieldb');
  var ctx = canvas.getContext('2d');
  var coordx = posJ*20;
  var coordy = posI*20;
  // console.log(posI, posJ, field[posI][posJ]);
  ctx.drawImage(man, coordx, coordy, 20, 20);
}

function containsPoint(point, list, step) {
  let path = list.slice(0, step);
  for (let i = 0; i < path.length; i++) {
    if (list[i].x === point.x && list[i].y === point.y) {
      return true;
    }
  }
  return false;
}

function drawScores(step) {
  var canvas = document.getElementById('fieldb');
  var ctx = canvas.getContext('2d');
  ctx.font = "10px Arial";

  if(step === undefined) step = 0;

  for(var i = 0; i < field.length; i++) {
    for(var j = 0; j < field[i].length; j++) {
      var value = field[i][j];
      var coordx = 20 * j + 5;
      var coordy = 20 * i + 15;
      if(value === ItemsEnum.WALL) continue;
      clear(i, j);

      if(containsPoint({x: j, y: i}, path, step)) {
        ctx.fillStyle = '#E5F77D';
        ctx.fillRect(20*j, 20*i, 20, 20);
        ctx.fillStyle = '#000000';
      }

      if(i === endi && j === endj) {
        ctx.fillStyle = '#F98948';
        ctx.fillRect(20*j, 20*i, 20, 20);
        ctx.fillStyle = '#000000';
      }

      ctx.fillText(value, coordx, coordy);
    }
  }
}

function drawStats() {
  $('#moves').text(totalMoves);
  $('#score').text(0);
}

function drawAllField(step) {
  drawElem(block, -1);
  drawScores(step);
  // draw player in front of everything
  drawPlayer(posi, posj);
}

function clear(posI, posJ) {
  var canvas = document.getElementById('fieldb');
  var ctx = canvas.getContext('2d');
  var coordx = 20*posJ;
  var coordy = 20*posI;
  ctx.clearRect(coordx, coordy, 20, 20);
}

function movement(where) {
  var moves = {
    left: {i: 0, j: -1},
    right: {i: 0, j: 1},
    up: {i: -1, j: 0},
    down: {i: 1, j: 0}
  };
  di = moves[where].i;
  dj = moves[where].j;
  if(posi+di < 0 || posi+di >= field.length) {
    return false;
  }
  if(posj+dj < 0 || posj+dj >= field[0].length) {
    return false;
  }

  return field[posi+di][posj+dj] !== ItemsEnum.WALL;
}

function doKeyDown(e) {
  //up
  if( e.keyCode == 38 ) {
    // with box
    if(movement("up")) {
      clear(posi, posj);
      posi -= 1;
    }
  }
  //left
  if ( e.keyCode == 37) {
    if(movement("left")) {
      clear(posi, posj);
      posj -= 1;
    }
  }
  //down
  if ( e.keyCode == 40 ) {
    if(movement("down")) {
      clear(posi, posj);
      posi += 1;
    }
  }
  //right
  if ( e.keyCode == 39 ) {
    if(movement("right")) {
      clear(posi, posj);
      posj += 1;
    }
  }
  drawAllField(path.length);
}


function draw() {

  var canvas = document.getElementById('fieldb');

  var ctx = canvas.getContext('2d');
  document.addEventListener("keydown", doKeyDown, true);

  // Create new img element
  man.src = '/static/img/man.png'; // Set source path
  block.src = '/static/img/border.png';
  cherry.src = '/static/img/cherry.png';

  man.onload = function() {
    drawPlayer(posi, posj);
  };

  block.onload = function() {
    drawElem(block, -1);
  };
}


$(function() {
  $('#solvebutton').click(function() {
    var aPath = path.slice();
    console.log(aPath);
    var i = 0, score = 0;
    var step = setInterval(function move() {
      if(i >= aPath.length) {
        clearInterval(step);
        return;
      }
      $('#moves').text(totalMoves-i);
      var coord = aPath[i];
      posi = coord.y;
      posj = coord.x;
      score += field[posi][posj];
      $('#score').text(score);
      drawAllField(i);
      i++;
    }, 300);
  });
});
