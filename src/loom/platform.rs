#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn spawn<O, F>(future: F) -> tokio::task::JoinHandle<O>
where
    O: Send + 'static,
    F: std::future::Future<Output = O> + Send + 'static,
{
    tokio::spawn(future)
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn spawn<F>(future: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(future);
}

#[cfg(not(target_arch = "wasm32"))]
pub type BoxFuture<'a, T> = futures::future::BoxFuture<'a, T>;
#[cfg(target_arch = "wasm32")]
pub type BoxFuture<'a, T> = futures::future::LocalBoxFuture<'a, T>;
