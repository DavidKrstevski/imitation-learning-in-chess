<script>
  export let moveRows = [];
</script>

<aside class="moves-panel">
  <h2>Letzte Züge</h2>
  <div class="moves-list">
    {#if moveRows.length === 0}
      <div class="move-row empty">Noch keine Züge</div>
    {:else}
      {#each moveRows as pair}
        <div class="move-row">
          <span class="move-no">{pair.number}.</span>
          <span class="move-white">{pair.white}</span>
          <span class="move-black">{pair.black}</span>
        </div>
      {/each}
    {/if}
  </div>
</aside>

<style>
  .moves-panel {
    min-width: 220px;
    border: 1px solid #c8c8c8;
    padding: 10px;
    background: #fff;
  }

  .moves-panel h2 {
    margin: 0 0 8px 0;
    font-size: 15px;
    color: #6b6b6b;
  }

  .moves-list {
    max-height: 430px;
    overflow-y: auto;
    font-size: 14px;
  }

  .move-row {
    display: grid;
    grid-template-columns: 36px 1fr 1fr;
    gap: 8px;
    padding: 3px 0;
    border-radius: 4px;
  }

  .move-row:nth-child(odd) {
    background: #f7f7f7;
  }

  .move-row.empty {
    display: block;
    color: #666;
  }

  @media (max-width: 900px) {
    .moves-panel {
      width: 100%;
      min-width: unset;
    }
  }
</style>
