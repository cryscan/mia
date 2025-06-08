#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn spawn<F>(future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
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
#[inline]
pub fn spawn_blocking<F, R>(f: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    tokio::task::spawn_blocking(f)
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn spawn_blocking<F, R>(f: F) -> R
where
    F: FnOnce() -> R + 'static,
    R: 'static,
{
    f()
}

#[cfg(not(target_arch = "wasm32"))]
pub type BoxFuture<'a, T> = futures::future::BoxFuture<'a, T>;
#[cfg(target_arch = "wasm32")]
pub type BoxFuture<'a, T> = futures::future::LocalBoxFuture<'a, T>;
