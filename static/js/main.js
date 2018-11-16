var ItemsEnum = {
  WALL: -1,
  EMPTY: 0,
  BOX: 1,
  MAN: 5
};

man = new Image();
block = new Image();
box = new Image();
boxCompleted = new Image();
cherry = new Image();
moves = 0;
cherriesLeft = 0;

var field, posi, posj;

$.get('/load_data', function(data) {
  console.log(data);
  field = data.field;
  posj = data.player.x;
  posi = data.player.y;
  draw();
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
  console.log(posI, posJ, field[posI][posJ]);
  ctx.drawImage(man, coordx, coordy, 20, 20);
}

function drawCherries() {
  var canvas = document.getElementById('fieldb');
  var ctx = canvas.getContext('2d');
  console.log(field);
  for(var i = 0; i < cherries.length; i++) {
    var x = cherries[i][1];
    var y = cherries[i][0];
    var coordx = 20*x;
    var coordy = 20*y;
    if (field[y][x] == ItemsEnum.BOX) {
      continue;
    }
    ctx.drawImage(cherry, coordx, coordy, 20, 20);
  }
}

function drawAllField() {
  drawElem(block, -1);

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
  drawAllField();
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