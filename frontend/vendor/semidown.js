(function (global) {
  "use strict";

  function MarkdownStreamChunker() {
    this.buffer = "";
    this.nextBlockId = 1;
    this.currentBlockId = null;
    this.listeners = {
      "block-start": [],
      "block-update": [],
      "block-end": [],
      end: [],
    };
  }

  MarkdownStreamChunker.prototype.write = function (chunk) {
    var data = this.buffer + (chunk || "");
    this.buffer = "";

    while (true) {
      var idx = data.indexOf("\n\n");
      if (idx === -1) break;

      var part = data.slice(0, idx);
      data = data.slice(idx + 2);

      this._emitUpdate(part);
      this._emitEnd();
    }

    if (data.length > 0) {
      this.buffer = data;
      this._emitUpdate(data);
    }
  };

  MarkdownStreamChunker.prototype.end = function () {
    if (this.buffer.length > 0) {
      this._emitUpdate(this.buffer);
      this.buffer = "";
      this._emitEnd();
    }
    this._emit("end");
  };

  MarkdownStreamChunker.prototype.on = function (event, fn) {
    this.listeners[event].push(fn);
  };

  MarkdownStreamChunker.prototype.off = function (event, fn) {
    this.listeners[event] = this.listeners[event].filter(function (listener) {
      return listener !== fn;
    });
  };

  MarkdownStreamChunker.prototype._emit = function (event, payload) {
    this.listeners[event].forEach(function (fn) {
      fn(payload);
    });
  };

  MarkdownStreamChunker.prototype._emitStart = function () {
    if (!this.currentBlockId) {
      this.currentBlockId = "block-" + this.nextBlockId++;
      this._emit("block-start", { blockId: this.currentBlockId });
    }
  };

  MarkdownStreamChunker.prototype._emitUpdate = function (content) {
    this._emitStart();
    this._emit("block-update", { blockId: this.currentBlockId, content: content });
  };

  MarkdownStreamChunker.prototype._emitEnd = function () {
    if (!this.currentBlockId) return;
    this._emit("block-end", { blockId: this.currentBlockId });
    this.currentBlockId = null;
  };

  function MarkdownParser() {
    if (!global.marked || typeof global.marked.parse !== "function") {
      throw new Error("marked is required for Semidown.");
    }
  }

  MarkdownParser.prototype.parse = function (markdown) {
    var html = global.marked.parse(markdown || "", { breaks: true });
    if (typeof global.profectusSanitizeHtml === "function") {
      html = global.profectusSanitizeHtml(html);
    }
    var fences = (markdown.match(/```/g) || []).length;
    return { html: html, isComplete: fences % 2 === 0 };
  };

  function HTMLRenderer(root) {
    this.root = root;
    this.blocks = new Map();
    this.root.innerHTML = "";
  }

  HTMLRenderer.prototype.createBlock = function (blockId) {
    var div = document.createElement("div");
    div.dataset.blockId = blockId;
    this.root.appendChild(div);
    this.blocks.set(blockId, div);
  };

  HTMLRenderer.prototype.updateBlock = function (blockId, html) {
    var el = this.blocks.get(blockId);
    if (el) {
      el.innerHTML = html;
    }
  };

  HTMLRenderer.prototype.finalizeBlock = function (blockId, isComplete) {
    var el = this.blocks.get(blockId);
    if (el && isComplete) {
      el.classList.add("md-block-complete");
    }
  };

  HTMLRenderer.prototype.clear = function () {
    this.root.innerHTML = "";
    this.blocks.clear();
  };

  function Semidown(targetElement) {
    this.chunker = new MarkdownStreamChunker();
    this.parser = new MarkdownParser();
    this.renderer = new HTMLRenderer(targetElement);
    this.blockContent = new Map();
    this.state = "idle";
    this._hookup();
  }

  Semidown.prototype.write = function (chunk) {
    if (this.state === "processing") {
      this.chunker.write(chunk);
    }
  };

  Semidown.prototype.end = function () {
    if (this.state === "processing") {
      this.chunker.end();
    }
  };

  Semidown.prototype.pause = function () {
    if (this.state === "processing") {
      this.state = "paused";
    }
  };

  Semidown.prototype.resume = function () {
    if (this.state === "paused") {
      this.state = "processing";
    }
  };

  Semidown.prototype.destroy = function () {
    this.state = "destroyed";
    this.renderer.clear();
    this.chunker.off("block-start", this._onBlockStart);
    this.chunker.off("block-update", this._onBlockUpdate);
    this.chunker.off("block-end", this._onBlockEnd);
    this.chunker.off("end", this._onEnd);
  };

  Semidown.prototype.getState = function () {
    return this.state;
  };

  Semidown.prototype._hookup = function () {
    var self = this;
    this.state = "processing";
    this._onBlockStart = function (p) {
      if (self.state !== "processing") return;
      self.renderer.createBlock(p.blockId);
    };
    this._onBlockUpdate = function (p) {
      if (self.state !== "processing") return;
      self.blockContent.set(p.blockId, p.content);
      var result = self.parser.parse(p.content);
      self.renderer.updateBlock(p.blockId, result.html);
    };
    this._onBlockEnd = function (p) {
      if (self.state !== "processing") return;
      var finalMD = self.blockContent.get(p.blockId) || "";
      var result = self.parser.parse(finalMD);
      self.renderer.finalizeBlock(p.blockId, result.isComplete);
      self.blockContent.delete(p.blockId);
    };
    this._onEnd = function () {
      self.state = "idle";
    };

    this.chunker.on("block-start", this._onBlockStart);
    this.chunker.on("block-update", this._onBlockUpdate);
    this.chunker.on("block-end", this._onBlockEnd);
    this.chunker.on("end", this._onEnd);
  };

  global.Semidown = Semidown;
})(window);
